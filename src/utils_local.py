import os
import time
import torch
import numpy as np
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from multiprocessing import Pool, cpu_count
# from peft.tuners.lora import Linear
from olora.tuners.lora import Linear
from tqdm import tqdm
import torch.distributed as dist


def cluster_module(module_info, n_clusters):
    name, weight = module_info
    hidden_dim = weight.shape[0]
    split_size = hidden_dim / n_clusters
    print(f"start weight clustering for {name}")
    kmeans = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=split_size,
        size_max=split_size,
        random_state=0
    ).fit(weight)
    labels = np.array([x for x in kmeans.labels_])
    print(f"weight clustering for {name}")
    return name, labels

## weight_cluster
def get_cluster_indices_multiprocessing(model, args):
    module_info = []
    cluster_indices = {}

    # Only proceed with clustering on rank 0
    if dist.get_rank() == 0:
        # Prepare data for multiprocessing
        module_info = [
            (name, F.normalize(module.weight.detach(), p=2, dim=0).cpu().numpy())
            for name, module in model.named_modules()
            if "loranew_B.default" in name
        ]
        print("Clustering step finished on rank 0!")

        # Create a pool of processes and map cluster_module function to each module
        with Pool(processes=cpu_count()) as pool:
            results = pool.starmap(cluster_module, [(info, args.n_clusters) for info in module_info])

        # Convert list of tuples to dictionary
        cluster_indices = {name: labels for name, labels in results}

    # Synchronize cluster_indices to all ranks
    for name, module in model.named_modules():
        if "loranew_B.default" in name:
            if dist.get_rank() == 0:
                tensor = torch.from_numpy(cluster_indices[name]).to('cuda')
            else:
                num_neurons = module.weight.size(0)
                tensor = torch.empty(num_neurons, dtype=torch.int32, device='cuda')
            dist.broadcast(tensor, src=0)
            cluster_indices[name] = tensor.cpu().numpy()

    return cluster_indices

## weight_cluster_combined
def get_cluster_indices_combined_multiprocessing(model, args):
    module_info = []
    cluster_indices = {}

    # Only proceed with clustering on rank 0
    if dist.get_rank() == 0:
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                # TODO merge
                combined_weight = module.weight + torch.matmul(module.loranew_A['default'].weight.T, module.loranew_B['default'].weight.T).T * module.scaling['default'] + torch.matmul(module.lora_A['default'].weight.T, module.lora_B['default'].weight.T).T * module.scaling['default']
                combined_weight = F.normalize(combined_weight.detach(), p=2, dim=0).cpu().numpy()
                module_info.append((name+".loranew_B.default", combined_weight))
        print("step one finished on rank 0!")
        # module_info = module_info[:2]
        # Create a pool of processes and map cluster_module function to each module
        start_time = time.time()  # Start time
        with Pool(processes=cpu_count()//2) as pool:  # Single process is found to be the fastest in experiments
            results = pool.starmap(cluster_module, [(info, args.n_clusters) for info in module_info])
        end_time = time.time()  # End time
        print(f"Clustering completed in {end_time - start_time:.2f} seconds on rank 0.")
        # Convert list of tuples to dictionary
        cluster_indices = {name: labels for name, labels in results}
    
    # Synchronize cluster_indices to all ranks
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            name = name + '.loranew_B.default'
            if dist.get_rank() == 0:
                tensor = torch.from_numpy(cluster_indices[name]).to('cuda')
            else:
                num_neurons = module.weight.size(0)
                tensor = torch.empty(num_neurons, dtype=torch.int32, device='cuda')
            dist.broadcast(tensor, src=0)
            cluster_indices[name] = tensor.cpu().numpy()

    return cluster_indices

## weight_cluster_combined
def get_cluster_indices_combined(model, args):
    module_info = []
    cluster_indices = {}

    # Only proceed with clustering on rank 0
    if dist.get_rank() == 0:
        for name, module in model.named_modules():
            if isinstance(module, Linear):
                # TODO merge
                combined_weight = module.weight + torch.matmul(module.lora_A['default'].weight.T, module.lora_B['default'].weight.T).T * module.scaling['default'] + torch.matmul(module.loranew_A['default'].weight.T, module.loranew_B['default'].weight.T).T * module.scaling['default'] ## 理论上 loranew_A 和loranew_B 的乘积是 0，不需要做加法
                combined_weight = F.normalize(combined_weight.detach(), p=2, dim=0).cpu().numpy()
                module_info.append((name + ".loranew_B.default", combined_weight))
        print("step one finished on rank 0!")
        # Create a pool of processes and map cluster_module function to each module
        results = []
        start_time = time.time()
        for info in module_info:
            name, labels = cluster_module(info, args.n_clusters)
            results.append((name, labels))
        end_time = time.time()
        print(f"Clustering completed in {end_time - start_time:.2f} seconds on rank 0.")
        # Convert list of tuples to dictionary
        cluster_indices = {name: labels for name, labels in results}

    # Synchronize cluster_indices to all ranks
    dist.barrier()
    for name, module in model.named_modules():
        if isinstance(module, Linear):
            name = name + '.loranew_B.default'
            if dist.get_rank() == 0:
                tensor = torch.from_numpy(cluster_indices[name]).to('cuda')
            else:
                num_neurons = module.loranew_B.default.weight.size(0)
                tensor = torch.empty(num_neurons, dtype=torch.int32, device='cuda')
            dist.broadcast(tensor, src=0)
            cluster_indices[name] = tensor.cpu().numpy()

    return cluster_indices

## co-activation
def cluster_graph(graph_info, n_clusters):
    name, co_activation_graph = graph_info
    print(f"start co-activation clustering for {name}")
    spectral_cluster = SpectralClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        assign_labels='discretize',
        random_state=42
    )
    labels = spectral_cluster.fit_predict(co_activation_graph)
    print(f"co-activation clustering for {name} completed")
    return name, labels

def get_cluster_indices_co_activation(trainer, model, dataloader, args):
    co_activation_graph_dict = {}
    hooks = []

    def hook_fn(module, input, output, name):
        hidden_dim = output.size(-1)
        accu_out = output.detach().reshape(-1, hidden_dim).unsqueeze(dim=-1)
        if co_activation_graph_dict[name] is not None:
            co_activation_graph_dict[name] += torch.bmm(accu_out, accu_out.transpose(1, 2)).abs().mean(0)
        else:
            co_activation_graph_dict[name] = torch.bmm(accu_out, accu_out.transpose(1, 2)).abs().mean(0)

    for name, module in model.named_modules():
        if isinstance(module, Linear):
            module._forward_hooks.clear()
            name = name + '.loranew_B.default'
            co_activation_graph_dict[name] = None
            hook = module.register_forward_hook(
                lambda module, input, output, name=name: hook_fn(module, input, output, name)
            )
            hooks.append(hook)
            print(f"register hooks for {name}")

    model.eval()
    processed_samples = 0
    max_samples = 500 / dist.get_world_size()
    for step, inputs in tqdm(enumerate(dataloader), total=len(dataloader)):
        inputs = trainer._prepare_inputs(inputs)
        with trainer.compute_loss_context_manager():
            trainer.compute_loss(model, inputs)
        # processed_samples += inputs['input_ids'].size(0)
        # if processed_samples > max_samples:
        #     break
    dist.barrier()

    for name, co_activation in co_activation_graph_dict.items():
        co_activation_tensor = co_activation.detach()
        # if dist.get_rank() == 0:
        #     print(co_activation_tensor)
        dist.reduce(co_activation_tensor, dst=0, op=dist.ReduceOp.SUM)
        # if dist.get_rank() == 0:
        #     print(co_activation_tensor)
        # exit()
        if dist.get_rank() == 0:
            co_activation_graph_dict[name] = co_activation_tensor.cpu().numpy()

    original_tokenizers_parallelism = os.getenv("TOKENIZERS_PARALLELISM")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    cluster_indices_dict = {}
    if dist.get_rank() == 0:
        # 经过实验，单进程是最快的
        with Pool(processes=1) as pool:
            results = pool.starmap(cluster_graph, [(info, args.n_clusters) for info in co_activation_graph_dict.items()])
        cluster_indices_dict = dict(results)

    for key, _ in co_activation_graph_dict.items():
        if dist.get_rank() == 0:
            tensor = torch.from_numpy(cluster_indices_dict[key]).to('cuda')
        else:
            tensor = torch.empty(co_activation_graph_dict[key].size(0), dtype=torch.int64, device='cuda')
        dist.broadcast(tensor, src=0)
        # if dist.get_rank() == 0:
        # print(dist.get_rank(),":", tensor, "\n")
        # exit()
        cluster_indices_dict[key] = tensor.cpu().numpy()
    
    if original_tokenizers_parallelism:
        os.environ["TOKENIZERS_PARALLELISM"] = original_tokenizers_parallelism
    else:
        del os.environ["TOKENIZERS_PARALLELISM"]

    # Remove hooks after computation
    for hook in hooks:
        hook.remove()

    return cluster_indices_dict

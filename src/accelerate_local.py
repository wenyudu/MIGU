import torch
import functools
import torch.distributed as dist
from accelerate.utils import DeepSpeedEngineWrapper

def get_module(root_module, module_name):
    """Retrieve a submodule from a root module based on the module's name."""
    attrs = module_name.split('.')
    lora_index = next(i for i, attr in enumerate(attrs) if 'lora' in attr)
    return functools.reduce(getattr, attrs[:lora_index+1], root_module)

def apply_importance_mask(name, module, importance_mask):
    """Apply the importance mask to the gradients of a module's default weights."""
    if hasattr(module, 'default') and hasattr(module.default, 'weight'):
        assert module.default.weight.grad is not None, f"{module} has no grad"
        if module.default.weight.grad is not None:
            module.default.weight.grad *= importance_mask.unsqueeze(dim=-1).to(module.default.weight.grad.device)

def compute_importance_mask(activation, ini_threshold, n_cluster, method, cluster_indice):
    """Compute the importance mask based on the provided method."""
    # activation, kwargs['ini_threshold'], kwargs['cluster_constructure_method'], cluster_indice
    device = activation[0].device
    hidden_dim = activation.shape[-1]
    # activation = torch.cat([item.reshape(-1, hidden_dim) for item in activation], dim=0)
    # importance = torch.mean(activation.abs(), dim=0)
    # dist.barrier()
    dist.all_reduce(activation, op=dist.ReduceOp.AVG)
    if method == 'sequential':
        activation_chunks = activation.chunk(n_cluster)
        activation = torch.stack([chunk.sum() for chunk in activation_chunks])
        threshold = torch.quantile(activation, ini_threshold)
        importance_mask = (activation > threshold).float().to(device)
        assert hidden_dim % n_cluster ==0, "hidden_dim must be divisible by n_cluster."
        importance_mask = importance_mask.repeat_interleave(hidden_dim // n_cluster)
    elif method == 'weight_cluster' or method == 'weight_cluster_combined' \
        or method == 'co-activation':
        # cluster_indice = torch.tensor(cluster_indice, dtype=torch.int64, device=device)
        # cluster_sums = torch.zeros(n_cluster, dtype=importance.dtype, device=device)
        # cluster_sums = cluster_sums.scatter_add(0, cluster_indice, importance)
        # threshold = torch.quantile(cluster_sums, ini_threshold)
        # cluster_mask = (cluster_sums > threshold).float().to(device)
        # importance_mask = torch.index_select(cluster_mask, 0, cluster_indice)

        cluster_indice = torch.tensor(cluster_indice, dtype=torch.int64, device=device)
        cluster_sums = torch.zeros(n_cluster, dtype=activation.dtype, device=device)
        cluster_sums.index_add_(0, cluster_indice, activation)
        importance_mask = cluster_sums[cluster_indice]
        # importance_mask = torch.zeros_like(importance)
        # for i in range(importance_mask.shape[0]):
        #     importance_mask[i] = cluster_sums[cluster_indice[i]]
        threshold = torch.quantile(importance_mask, ini_threshold)
        importance_mask = (importance_mask > threshold).float().to(device)
    else:
        threshold = torch.quantile(activation, ini_threshold)
        importance_mask = (activation > threshold).float().to(device)
    return importance_mask

def backward(self, loss, **kwargs):
    """Custom backward method that applies importance masks to gradients."""
    self.engine.backward(loss)

    if not kwargs.get('is_first_task', True) and (kwargs.get('method') == "cluster_activate" or kwargs.get('method') == "random_update")\
        and self.engine.is_gradient_accumulation_boundary():

        ## parse parameters
        method = kwargs['method']
        activations = kwargs['activation']
        n_clusters = kwargs.get('n_clusters')
        ini_threshold = kwargs.get('ini_threshold')
        cluster_indice_dict = kwargs.get('cluster_indice', {})
        activation_combined = kwargs.get('activation_combined', False)
        cluster_constructure_method = kwargs['cluster_constructure_method']

        if method == "cluster_activate":
            for name, activation in activations.items():
                if "input" in name:
                    continue

                if "loranew_A" in name:
                    # activation, ini_threshold, n_cluster, method, cluster_indice
                    importance_mask = compute_importance_mask(activation, ini_threshold, None, None, None)
                    module_name = name
                elif ("loranew_B" not in name) == activation_combined:
                    module_name = name + ".loranew_B.default" if activation_combined else name
                    if cluster_constructure_method == "sequential":
                        importance_mask = compute_importance_mask(activation, ini_threshold, n_clusters,
                                                                  cluster_constructure_method, None)
                    else:
                        if module_name not in cluster_indice_dict:
                            print(f"lack of cluster_indice for {module_name}")
                            break
                        cluster_indice = cluster_indice_dict[module_name]
                        importance_mask = compute_importance_mask(activation, ini_threshold, n_clusters,
                                                                  cluster_constructure_method, cluster_indice)

                module = get_module(self.engine, module_name)
                apply_importance_mask(name, module, importance_mask)
        elif method == "random_update":
            for name, module in self.engine.named_modules():
                if hasattr(module, 'default') and hasattr(module.default, 'weight'):
                    assert module.default.weight.grad is not None, f"{module} has no grad"
                    cur_dim = module.default.weight.grad.shape[0]
                    importance_mask = torch.ones(cur_dim, dtype=torch.float)
                    ones_indices = torch.randperm(cur_dim)[:int(cur_dim*ini_threshold)]
                    importance_mask[ones_indices] = 0.0
                    module.default.weight.grad *= importance_mask.unsqueeze(dim=-1).to(module.default.weight.grad.device)

    self.engine.step()

def replace_accelerator_backward_with_own_backward():
    """Replace the Accelerate library's backward method with the custom backward method."""
    DeepSpeedEngineWrapper.backward = backward

import json
import random
import os
import shutil
# JSON文件的路径
source_root_path = '/lustre/S/zhangyang/chengshuang/CL/cluster_activate_lora_rehearsal/CL_Benchmark/'
all_entries = os.listdir(source_root_path)
task_categories = [entry for entry in all_entries if os.path.isdir(os.path.join(source_root_path, entry))]


for task_catogory in task_categories:
    task_category_path = os.path.join(source_root_path, task_catogory)
    data_categories = os.listdir(task_category_path)
    data_categories = [entry for entry in data_categories if "sampling" not in entry]
    for category in data_categories:
        category_path = os.path.join(task_category_path, category)
        source_file_path = f"{category_path}/train.json"
        source_label_path = f"{category_path}/labels.json"
        with open(source_file_path, 'r') as file:
            data = json.load(file)
        # print(category, ":", len(data))
        # if len(data) < 200:
        #     raise ValueError("The list in the JSON file contains less than 200 dictionaries.")
        # 输出文件的目录，以及采样次数
        for i in range(2, 5):
            destination_root_path = f"{category_path}_sampling/{i}"
            destination_file_path = f"{destination_root_path}/train.json"
            destination_label_path = f"{destination_root_path}/labels.json"

            # 确保输出目录存在
            if not os.path.exists(destination_root_path):
                os.makedirs(destination_root_path)

            ## 首先是 label 文件
            shutil.copy(source_label_path, destination_label_path)  # 仅复制内容和权限
            
            ## 然后是 train.json
            num_sample = int(len(data) * 0.02)
            selected_dicts = random.sample(data, num_sample)

            # 写入到输出文件
            with open(destination_file_path, 'w') as file:
                json.dump(selected_dicts, file, indent=4, ensure_ascii=False)

            print(f"{num_sample} random dictionaries have been written to {destination_file_path}")


# import os
# import shutil

# def delete_sampling_directories(root_path):
#     # 遍历文件夹
#     for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
#         # topdown=False表示从底层子文件夹开始检查
#         for dirname in dirnames:
#             # 构建完整的文件夹路径
#             full_dir_path = os.path.join(dirpath, dirname)
#             # 检查文件夹名称是否包含'sampling'
#             if 'sampling' in dirname:
#                 # 删除包含'sampling'的文件夹
#                 shutil.rmtree(full_dir_path)
#                 print(f"Deleted: {full_dir_path}")

# # 指定顶层文件夹路径
# top_folder = '/lustre/S/zhangyang/chengshuang/CL/cluster_activate_lora_rehearsal/CL_Benchmark'

# # 删除所有名称包含'sampling'的文件夹
# delete_sampling_directories(top_folder)

#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"


port=$(shuf -i25000-30000 -n1)
method="cluster_activate"
model="t5_large"

# CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port $port src/run_uie_lora.py \
#    --do_train \
#    --do_predict \
#    --predict_with_generate \
#    --model_name_or_path output/${model}/${method}/order_1/outputs/2-amazon/adapter \
#    --data_dir CL_Benchmark \
#    --task_config_dir configs/order1_configs/yahoo \
#    --instruction_file configs/instruction_config.json \
#    --instruction_strategy single \
#    --output_dir output/${model}/${method}/order_1/outputs/3-yahoo \
#    --per_device_train_batch_size 32 \
#    --per_device_eval_batch_size 128 \
#    --gradient_accumulation_steps 1 \
#    --learning_rate 1e-03 \
#    --num_train_epochs 1 \
#    --deepspeed configs/ds_configs/stage0.config \
#    --run_name order1_round2 \
#    --max_source_length 512 \
#    --max_target_length 50 \
#    --generation_max_length 50 \
#    --add_task_name True \
#    --add_dataset_name True \
#    --overwrite_output_dir \
#    --overwrite_cache \
#    --lr_scheduler_type constant \
#    --warmup_steps 0 \
#    --logging_strategy steps \
#    --logging_steps 10 \
#    --evaluation_strategy no \
#    --save_strategy no \
#    --save_steps 1500 \
#    --lamda_1 0.5 \
#    --lamda_2 0 \
#    --method ${method} \
#    --is_first_task False \
#    --n_clusters 32 \
#    --cluster_constructure_method "sequential" \
#    --ini_threshold 0.5 \
#    --activation_combined True

method="cluster_activate"
CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/order_1/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order1_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/order_1/outputs/3-yahoo \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round3 \
   --max_source_length 512 \
   --max_target_length 50 \
   --generation_max_length 50 \
   --add_task_name True \
   --add_dataset_name True \
   --overwrite_output_dir \
   --overwrite_cache \
   --lr_scheduler_type constant \
   --warmup_steps 0 \
   --logging_strategy steps \
   --logging_steps 10 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters 32 \
   --cluster_constructure_method "co-activation" \
   --ini_threshold 0.5 \
   --activation_combined False

   # --cluster_constructure_method "weight_cluster" \
# # CUDA_VISIBLE_DEVICES=0,1 deepspeed --master_port $port src/run_uie_lora.py \
# #    --do_train \
# #    --do_predict \
# #    --predict_with_generate \
# #    --model_name_or_path output/${model}/${method}/order_1/outputs/3-yahoo/adapter \
# #    --data_dir CL_Benchmark \
# #    --task_config_dir configs/order1_configs/agnews \
# #    --instruction_file configs/instruction_config.json \
# #    --instruction_strategy single \
# #    --output_dir output/${model}/${method}/order_1/outputs/4-agnews \
# #    --per_device_train_batch_size 32 \
# #    --per_device_eval_batch_size 128 \
# #    --gradient_accumulation_steps 1 \
# #    --learning_rate 1e-03 \
# #    --num_train_epochs 1 \
# #    --deepspeed configs/ds_configs/stage0.config \
# #    --run_name order1_round4 \
# #    --max_source_length 512 \
# #    --max_target_length 50 \
# #    --generation_max_length 50 \
# #    --add_task_name True \
# #    --add_dataset_name True \
# #    --overwrite_output_dir \
# #    --overwrite_cache \
# #    --lr_scheduler_type constant \
# #    --warmup_steps 0 \
# #    --logging_strategy steps \
# #    --logging_steps 10 \
# #    --evaluation_strategy no \
# #    --save_strategy no \
# #    --save_steps 1500 \
# #    --lamda_1 0.5 \
# #    --lamda_2 0 \
# #    --method ${method} \
# #    --is_first_task False \
# #    --n_clusters 32 \
# #    --cluster_constructure_method "sequential" \
# #    --ini_threshold 0.5 \
# #    --activation_combined False

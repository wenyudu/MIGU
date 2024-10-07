#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)

model=$1
tuning_method=$2
method=$3
ini_threshold=$4
seed=$5

deepspeed --master_port $port src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path initial_model/${model} \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_2/outputs/1-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order2_round1 \
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
   --lamda 0 \
   --method ${method} \
   --is_first_task True \
   --seed=${seed}

sleep 5

deepspeed --master_port $port src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_2/outputs/1-dbpedia/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_2/outputs/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order2_round2 \
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
   --lamda 0 \
   --is_first_task False \
   --method ${method} \
   --ini_threshold ${ini_threshold} \
   --seed=${seed}

sleep 5

deepspeed --master_port $port src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_2/outputs/2-amazon/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_2/outputs/3-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order2_round3 \
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
   --lamda 0 \
   --is_first_task False \
   --method ${method} \
   --ini_threshold ${ini_threshold} \
   --seed=${seed}

sleep 5

deepspeed --master_port $port src/run_uie_ft.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_2/outputs/3-agnews/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order2_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_2/outputs/4-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate 1e-04 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order2_round4 \
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
   --lamda 0 \
   --is_first_task False \
   --method ${method} \
   --ini_threshold ${ini_threshold} \
   --seed=${seed}

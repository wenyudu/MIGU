#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)

model=$1
tuning_method=$2
method=$3
ini_threshold=$4
seed=$5
lr=1e-4
# yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag → yahoo → multirc → boolqa → wic

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path initial_model/${model} \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/1-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round1 \
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
   --seed ${seed}

sleep 5

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/1-yelp/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round2 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/2-amazon/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/MNLI \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/3-MNLI \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 2 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round3 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/3-MNLI/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/CB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/4-CB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round4 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/4-CB/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/COPA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/5-COPA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round5 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/5-COPA/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/QQP \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/6-QQP \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round6 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/6-QQP/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/RTE \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/7-RTE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round7 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/7-RTE/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/IMDB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/8-IMDB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round8 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/8-IMDB/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/SST-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/9-SST-2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round9 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/9-SST-2/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/10-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round10 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/10-dbpedia/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/11-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round11 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/11-agnews/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/12-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round12 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/12-yahoo/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/MultiRC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/13-MultiRC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round13 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/13-MultiRC/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/BoolQA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/14-BoolQA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round14 \
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

deepspeed --master_port 25000 src/run_uie_ft.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${tuning_method}/${method}/order_6/outputs/14-BoolQA/tuning_weight \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/WiC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${tuning_method}/${method}/order_6/outputs/15-WiC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 1 \
   --learning_rate ${lr} \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name long_round15 \
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
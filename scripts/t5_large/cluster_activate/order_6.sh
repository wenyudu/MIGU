#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"

port=$(shuf -i25000-30000 -n1)
method="cluster_activate"
model="t5_large"
cluster=$1
ini_threshold=$2
cluster_constructure_method=$3
activation_combined=$4
# yelp → amazon → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag → yahoo → multirc → boolqa → wic
 
# bash scripts/order_6.sh > output/${model}/${method}/${cluster_constructure_method}/order_6/logs/train_and_infer.log 2>&1

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/${model} \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/1-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task True

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/1-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/2-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/2-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/MNLI \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/3-MNLI \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/3-MNLI/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/CB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/4-CB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined} 

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/4-CB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/COPA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/5-COPA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0.1 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/5-COPA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/QQP \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/6-QQP \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0.1 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/6-QQP/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/RTE \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/7-RTE \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0.3 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/7-RTE/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/IMDB \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/8-IMDB \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined} 

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/8-IMDB/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/SST-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/9-SST-2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 0.5 \
   --lamda_2 0.1 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/9-SST-2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/10-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/10-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/11-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/11-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/12-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 5 \
   --lamda_2 0.1 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/12-yahoo/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/MultiRC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/13-MultiRC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 5 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/13-MultiRC/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/BoolQA \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/14-BoolQA \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 5 \
   --lamda_2 0.1 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5

deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/14-BoolQA/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order6_configs/WiC \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_6/outputs/15-WiC \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
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
   --lamda_1 5 \
   --lamda_2 0.3 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}
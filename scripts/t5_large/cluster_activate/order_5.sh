#!/bin/bash
set -x

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
# export TRANSFORMERS_CACHE=/root/.cache/huggingface
# multirc → boolqa → wic → mnli → cb → copa → qqp → rte → imdb → sst-2 → dbpedia → ag → yelp → amazon → yahoo
port=$(shuf -i25000-30000 -n1)
method="cluster_activate"
model="t5_large"
cluster=$1
ini_threshold=$2
cluster_constructure_method=$3
activation_combined=$4
# bash scripts/order_5.sh > logs_and_outputs/order_5/logs/train_and_infer.log 2>&1
# 1-multirc
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/${model} \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/multirc \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/1-multirc \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 2-boolqa
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/1-multirc/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/boolqa \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/2-boolqa \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 3-wic
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/2-boolqa/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/wic \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/3-wic \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 4-mnli
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/3-wic/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/mnli \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/4-mnli \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 5-cb
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/4-mnli/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/cb \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/5-cb \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 6-copa
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/5-cb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/copa \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/6-copa \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 7-qqp
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/6-copa/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/qqp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/7-qqp \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 8-rte
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/7-qqp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/rte \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/8-rte \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 9-imdb
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/8-rte/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/imdb \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/9-imdb \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 10-sst-2
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/9-imdb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/sst-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/10-sst-2 \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 11-dbpedia
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/10-sst-2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/11-dbpedia \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 12-ag
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/11-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/12-agnews \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

#13-yelp
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/12-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/13-yelp \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 14-amazon
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/13-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/14-amazon \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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

# 15-yahoo
deepspeed --master_port 25000 src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/14-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order5_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_5/outputs/15-yahoo \
   --per_device_train_batch_size 32 \
   --per_device_eval_batch_size 128 \
   --gradient_accumulation_steps 1 \
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
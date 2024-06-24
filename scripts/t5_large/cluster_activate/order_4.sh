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
# mnli → cb → wic → copa → qqp → boolqa → rte → imdb → yelp → amazon → sst-2 → dbpedia → ag → multirc → yahoo
# 1-mnli
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path initial_model/${model} \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/mnli \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/1-mnli \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order4_round1 \
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
   --logging_steps 3 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0 \
   --lamda_2 0 \
   --method ${method} \
   --is_first_task True

sleep 5
# 2-cb
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/1-mnli/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/cb \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/2-cb \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round2 \
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
   --logging_steps 3 \
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
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/2-cb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/wic \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/3-wic \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order4_round3 \
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
   --logging_steps 3 \
   --evaluation_strategy no \
   --save_strategy no \
   --save_steps 1500 \
   --lamda_1 0.5 \
   --lamda_2 0 \
   --lora_dim 8 \
   --method ${method} \
   --is_first_task False \
   --n_clusters $cluster \
   --cluster_constructure_method ${cluster_constructure_method} \
   --ini_threshold ${ini_threshold} \
   --activation_combined ${activation_combined}

sleep 5
# 4-copa
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/3-wic/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/copa \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/4-copa \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round4 \
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
   --logging_steps 3 \
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
# 5-qqp
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/4-copa/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/qqp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/5-qqp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round2 \
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
   --logging_steps 3 \
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
# 6-boolqa
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/5-qqp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/boolqa \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/6-boolqa \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
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
   --logging_steps 3 \
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
# 7-rte
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/6-boolqa/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/rte \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/7-rte \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round4 \
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
   --logging_steps 3 \
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
# 8-imdb
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/7-rte/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/imdb \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/8-imdb \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round2 \
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
   --logging_steps 3 \
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
# 9-yelp
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/8-imdb/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/yelp \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/9-yelp \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
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
   --logging_steps 3 \
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
# 10- amazon
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/9-yelp/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/amazon \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/10-amazon \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round4 \
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
   --logging_steps 3 \
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
# 11-sst-2
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/10-amazon/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/sst-2 \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/11-sst-2 \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round2 \
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
   --logging_steps 3 \
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
# 12-dbpedia
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/11-sst-2/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/dbpedia \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/12-dbpedia \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
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
   --logging_steps 3 \
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
# 13-ag
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/12-dbpedia/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/agnews \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/13-agnews \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round4 \
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
   --logging_steps 3 \
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
# 14-multirc
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/13-agnews/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/multirc \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/14-multirc \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
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
   --logging_steps 3 \
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
deepspeed --master_port $port src/run_uie_lora.py \
   --do_train \
   --do_predict \
   --predict_with_generate \
   --model_name_or_path output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/14-multirc/adapter \
   --data_dir CL_Benchmark \
   --task_config_dir configs/order4_configs/yahoo \
   --instruction_file configs/instruction_config.json \
   --instruction_strategy single \
   --output_dir output/${model}/${method}/${cluster_constructure_method}/order_4/outputs/15-yahoo \
   --per_device_train_batch_size 8 \
   --per_device_eval_batch_size 64 \
   --gradient_accumulation_steps 4 \
   --learning_rate 1e-03 \
   --num_train_epochs 1 \
   --deepspeed configs/ds_configs/stage0.config \
   --run_name order1_round4 \
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
   --logging_steps 3 \
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

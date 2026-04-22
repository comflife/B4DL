#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0,1 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = global batch size
MASTER_PORT=29571

IFS=',' read -r -a GPU_LIST <<< "$gpu_vis"
NUM_GPUS=${#GPU_LIST[@]}

CUDA_VISIBLE_DEVICES=$gpu_vis torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT vtimellm/train/train.py \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version plain \
    --data_path ./lidarllm_only_dataset/stage1_train_converted.json \
    --feat_folder ./lidarclip/stage1_features \
    --tune_mm_mlp_adapter True \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage1 \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    "$@"

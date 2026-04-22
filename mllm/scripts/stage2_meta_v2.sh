#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0,1,2,3 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = global batch size
MASTER_PORT=29576

IFS=',' read -r -a GPU_LIST <<< "$gpu_vis"
NUM_GPUS=${#GPU_LIST[@]}

CUDA_VISIBLE_DEVICES=$gpu_vis torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT vtimellm/train/train.py \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_combined_meta_v2.json \
    --feat_folder ./lidarclip/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2-meta-v2 \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "no" \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --lambda_reg 1.0 \
    --use_ntl_loss \
    --lambda_ntl 0.5 \
    --use_temporal_embedding \
    "$@"

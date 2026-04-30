#!/bin/bash
# Joint LidarCLIP + VTimeLLM stage2 fine-tune on 3dtesting (Nu-Grounding).
# - LidarCLIP-SST is loaded from pretrained ckpt and trained jointly (not frozen).
# - mm_projector loaded from stage1 (and frozen by default below).
# - LoRA on Vicuna.
#
# Memory: SST + 7B Vicuna LoRA + raw point clouds is heavy. We default to
# per_device_batch=2 with grad-accum=16 (effective 32) and gradient_checkpointing=True.

MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=3

# torch 1.9 + setuptools>=60 compat: force stdlib distutils so torch.utils.tensorboard
# can read distutils.version (intercepted by setuptools-shim otherwise).
export SETUPTOOLS_USE_DISTUTILS=stdlib

# Pin to lidarclip conda env's python (system /usr/local/python3.10 lacks torch).
PYTHON=/home/byounggun/anaconda3/envs/lidarclip/bin/python

CUDA_VISIBLE_DEVICES=$gpu_vis $PYTHON vtimellm/train/train_3dtesting.py \
    --lora_enable True \
    --model_name_or_path ../base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ../3dtesting_dataset/train.json \
    --pc_index_path ../3dtesting_dataset/sample_token_to_lidar.json \
    --nuscenes_root ../nuscenes \
    --max_points 40000 \
    --pretrain_mm_mlp_adapter ../mllm/checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --lidar_sst_config /home/byounggun/B4DL/encoders/lidarclip/lidarclip/model/sst_encoder_only_config.py \
    --lidar_ckpt /home/byounggun/B4DL/encoders/lidarclip/ckpt/lidarclip_mm/epochepoch=04.ckpt \
    --freeze_lidar False \
    --output_dir ../mllm/checkpoints/vtimellm-$MODEL_VERSION-3dtesting-joint \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 4 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 128 \
    --lora_alpha 256 \
    --weight_decay 0.01 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    "$@"

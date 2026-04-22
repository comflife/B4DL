#!/bin/bash

set -e

MODEL_VERSION=vicuna-v1-5-7b
GPU_VIS=0,1
MASTER_PORT_STAGE1=29571
MASTER_PORT_STAGE2=29575

# Parse IFS for GPU count
IFS=',' read -r -a GPU_LIST <<< "$GPU_VIS"
NUM_GPUS=${#GPU_LIST[@]}

echo "============================================================"
echo "Running Stage 1 + Stage 2 sequentially on GPUs: $GPU_VIS"
echo "============================================================"

# ==================== STAGE 1 ====================
echo ""
echo "[Stage 1] Starting training..."
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_VIS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port $MASTER_PORT_STAGE1 \
    vtimellm/train/train.py \
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

# Check Stage 1 output
STAGE1_OUTPUT="./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin"

if [ ! -f "$STAGE1_OUTPUT" ]; then
    echo ""
    echo "[ERROR] Stage 1 output not found: $STAGE1_OUTPUT"
    echo "[ERROR] Stage 2 cannot start. Aborting."
    exit 1
fi

echo ""
echo "[Stage 1] ✅ Completed successfully."
echo "[Stage 1] Output found: $STAGE1_OUTPUT"

# ==================== STAGE 2 ====================
echo ""
echo "[Stage 2] Starting training..."
echo "============================================================"

CUDA_VISIBLE_DEVICES=$GPU_VIS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port $MASTER_PORT_STAGE2 \
    vtimellm/train/train.py \
    --lora_enable True \
    --model_name_or_path ./base_model/vicuna-v1-5-7b \
    --version v1 \
    --data_path ./b4dl_dataset/stage2_combined_meta.json \
    --feat_folder ./lidarclip/stage2_features \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-$MODEL_VERSION-stage1/mm_projector.bin \
    --output_dir ./checkpoints/vtimellm-$MODEL_VERSION-stage2 \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
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
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    "$@"

echo ""
echo "============================================================"
echo "Stage 1 + Stage 2 training completed successfully!"
echo "============================================================"
echo "Stage 1 checkpoint: ./checkpoints/vtimellm-$MODEL_VERSION-stage1/"
echo "Stage 2 checkpoint: ./checkpoints/vtimellm-$MODEL_VERSION-stage2/"

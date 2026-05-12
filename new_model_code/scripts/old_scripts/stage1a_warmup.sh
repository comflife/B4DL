#!/bin/bash
# Stage 1a — Projection-layer warmup for Qwen3-0.6B + SMAP.
# Freezes the LLM and only trains `mm_projector` (and the rows of the
# embedding/lm_head we just added — but we currently freeze those too; the
# projector alone is enough to pre-align SMAP features into Qwen's hidden
# space). High LR, 1 epoch on the combined nuCaption + nuGrounding data.
set -e

source /home/byounggun/anaconda3/etc/profile.d/conda.sh
conda activate /data1/byounggun/conda_envs/qwen_mm

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEAT_DIR="/data1/byounggun/3davs_b4dl/features/smap_lidar12"
OUT_DIR="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1a_q3d"
DATA_PATH="/data1/byounggun/3davs_b4dl/data/stage1_combined_q3d.json"
BASE_MODEL="Qwen/Qwen3-0.6B"

GPUS="${GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29581}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "$OUT_DIR"
cd "$NEW_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port $MASTER_PORT \
    train_qwen_sft.py \
    --model_name_or_path "$BASE_MODEL" \
    --mm_input_dim 512 \
    --tune_mm_only True \
    --coord_aux_weight 0.0 \
    --data_path "$DATA_PATH" \
    --feat_folder "$FEAT_DIR" \
    --max_length 2048 \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --eval_strategy "no" \
    --save_strategy "no" \
    --remove_unused_columns False \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 20 \
    --tf32 True \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name qwen_stage1a_q3d_warmup \
    "$@"

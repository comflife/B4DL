#!/bin/bash
# Stage 1b — Full FT on VoxelNeXt features.
#
# Loads the VoxelNeXt-stage-1a warmed-up projector pair and continues
# full-parameter fine-tuning, exactly like the SMAP version, but with
# K=256 spatial tokens per sample so the LLM finally has a real spatial
# signal to attend to for det_area localisation.
set -e

source /home/byounggun/anaconda3/etc/profile.d/conda.sh
conda activate /data1/byounggun/conda_envs/qwen_mm

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEAT_DIR="/data1/byounggun/3davs_b4dl/features/voxelnext_q256"
OUT_DIR="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_vxnxt"
DATA_PATH="/data1/byounggun/3davs_b4dl/data/stage1_combined_q3d.json"
BASE_MODEL="Qwen/Qwen3-0.6B"
PRETRAIN_PROJ="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1a_vxnxt/mm_projector.bin"

GPUS="${GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29582}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

if [ ! -f "$PRETRAIN_PROJ" ]; then
    echo "[error] missing warmed-up projector at $PRETRAIN_PROJ" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
cd "$NEW_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port $MASTER_PORT \
    train_qwen_sft.py \
    --model_name_or_path "$BASE_MODEL" \
    --mm_input_dim 128 \
    --mm_pos_dim 3 \
    --pretrain_mm_projector "$PRETRAIN_PROJ" \
    --tune_mm_only False \
    --coord_aux_weight 0.5 \
    --data_path "$DATA_PATH" \
    --feat_folder "$FEAT_DIR" \
    --max_length 2048 \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --remove_unused_columns False \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name qwen_stage1b_vxnxt \
    "$@"

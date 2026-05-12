#!/bin/bash
# Stage 1c — det_area-only continued SFT.
#
# Eval after Stage 1b shows the model has learned class echo + box copy
# (det_object) but failed to learn true visual grounding from text
# (det_area Car BEV mIoU 0.56% vs LiDAR-LLM 14.3%). Most predictions
# collapse to a constant "average box". The fix is to expose the model to
# more det_area-only practice on top of the stage-1b base, where echo
# samples can no longer dilute the localisation signal.
#
# Loads the stage-1b full-FT checkpoint (model + projector + tokenizer),
# trains for 2 more epochs on the 82k det_area subset, with the Q3D
# coord-aux L1 turned up to 1.0 so the differentiable distance gradient
# carries even more weight than during stage 1b.
set -e

source /home/byounggun/anaconda3/etc/profile.d/conda.sh
conda activate /data1/byounggun/conda_envs/qwen_mm

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FEAT_DIR="/data1/byounggun/3davs_b4dl/features/smap_lidar12"
OUT_DIR="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1c_detarea_q3d"
DATA_PATH="/data1/byounggun/3davs_b4dl/data/stage1c_detarea_q3d.json"
# Stage 1c starts from the stage-1b full-FT checkpoint, NOT from base
# Qwen3 — that's the whole point.
SFT_INIT="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_q3d"

GPUS="${GPUS:-0,1}"
MASTER_PORT="${MASTER_PORT:-29583}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

mkdir -p "$OUT_DIR"
cd "$NEW_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPUS torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port $MASTER_PORT \
    train_qwen_sft.py \
    --model_name_or_path "$SFT_INIT" \
    --mm_input_dim 512 \
    --tune_mm_only False \
    --coord_aux_weight 1.0 \
    --data_path "$DATA_PATH" \
    --feat_folder "$FEAT_DIR" \
    --max_length 1024 \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --remove_unused_columns False \
    --save_strategy "steps" \
    --save_steps 2500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb \
    --run_name qwen_stage1c_detarea \
    "$@"

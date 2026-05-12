#!/bin/bash
# Stage 1a — Projector warmup (frozen LLM) with VoxelNeXt + 999-bin coord tokens.
set -e

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate "$QWEN_MM_ENV"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$DATA_ROOT/checkpoints/qwen_stage1a_999"
DATA_PATH="$DATA_ROOT/data/stage1_combined_999.json"
BASE_MODEL="Qwen/Qwen3.5-9B"

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
    --model_name_or_path "$BASE_MODEL" \
    --mm_input_dim 128 \
    --mm_pos_dim 3 \
    --voxelnext_root "$VOXELNEXT_ROOT" \
    --voxelnext_ckpt "$VOXELNEXT_CKPT" \
    --voxelnext_top_k 256 \
    --voxelnext_freeze True \
    --tune_mm_only True \
    --data_path "$DATA_PATH" \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version v1.0-trainval \
    --n_sweeps 10 \
    --max_length 2048 \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --eval_strategy "no" \
    --remove_unused_columns False \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 2 \
    --learning_rate 1e-3 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name qwen_stage1a_999 \
    "$@"

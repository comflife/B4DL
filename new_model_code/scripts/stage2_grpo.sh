#!/bin/bash
# Stage 2 — GRPO fine-tuning from stage1c.
set -e

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate "$QWEN_MM_ENV"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/0526040099_A/3davs_b4dl}"
OUT_DIR="$DATA_ROOT/checkpoints/qwen_stage2_grpo_999"
SFT_MODEL="$DATA_ROOT/checkpoints/qwen_stage1c_999"
GRPO_DATA="$DATA_ROOT/data/grpo_train_999.json"

GPU="${GPUS:-0}"

if [ ! -d "$SFT_MODEL" ]; then
    echo "[error] missing stage1c checkpoint at $SFT_MODEL — run stage1c first" >&2
    exit 2
fi

if [ ! -f "$GRPO_DATA" ]; then
    echo "[error] missing GRPO training data at $GRPO_DATA — generate from det_area set first" >&2
    exit 2
fi

mkdir -p "$OUT_DIR"
cd "$NEW_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPU python train_qwen_grpo.py \
    --model_name_or_path "$SFT_MODEL" \
    --data_path "$GRPO_DATA" \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --weight_decay 0.0 \
    --warmup_steps 100 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name qwen_stage2_grpo_999 \
    "$@"

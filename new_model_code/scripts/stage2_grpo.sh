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
OUT_DIR="$DATA_ROOT/checkpoints/qwen_stage2_grpo_bevq_999"
SFT_MODEL="$DATA_ROOT/checkpoints/qwen_stage1c_bevq_999"
GRPO_DATA="$DATA_ROOT/data/grpo_train_999.json"
VOXELNEXT_TOKEN_MODE="${VOXELNEXT_TOKEN_MODE:-bev_query}"
BEV_QUERY_NUM="${BEV_QUERY_NUM:-576}"
BEV_QUERY_LAYERS="${BEV_QUERY_LAYERS:-2}"
BEV_QUERY_HEADS="${BEV_QUERY_HEADS:-8}"
BEV_MEMORY_MAX_TOKENS="${BEV_MEMORY_MAX_TOKENS:-0}"

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
    --sft_dir "$SFT_MODEL" \
    --data_path "$GRPO_DATA" \
    --output_dir "$OUT_DIR" \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version v1.0-trainval \
    --n_sweeps 10 \
    --voxelnext_root "$VOXELNEXT_ROOT" \
    --voxelnext_ckpt "$VOXELNEXT_CKPT" \
    --voxelnext_top_k 256 \
    --voxelnext_token_mode "$VOXELNEXT_TOKEN_MODE" \
    --bev_query_num "$BEV_QUERY_NUM" \
    --bev_query_layers "$BEV_QUERY_LAYERS" \
    --bev_query_heads "$BEV_QUERY_HEADS" \
    --bev_query_use_view_embed True \
    --bev_memory_max_tokens "$BEV_MEMORY_MAX_TOKENS" \
    --epochs 1 \
    --num_rollouts 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-6 \
    --log_every 10 \
    --save_every 1000 \
    "$@"

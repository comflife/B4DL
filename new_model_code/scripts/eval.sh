#!/bin/bash
# Generic eval on 3DTesting val set.
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_dir> [output_dir] [gpu_id]" >&2
    exit 1
fi

CKPT_DIR="$1"
shift

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    OUT_DIR="$1"
    shift
else
    OUT_DIR="$CKPT_DIR/eval_val"
fi

if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    GPU="$1"
    shift
else
    GPU=0
fi

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate "$QWEN_MM_ENV"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/0526040099_A/3davs_b4dl}"
VOXELNEXT_TOKEN_MODE="${VOXELNEXT_TOKEN_MODE:-bev_query}"
BEV_QUERY_NUM="${BEV_QUERY_NUM:-576}"
BEV_QUERY_LAYERS="${BEV_QUERY_LAYERS:-2}"
BEV_QUERY_HEADS="${BEV_QUERY_HEADS:-8}"
BEV_QUERY_DIM="${BEV_QUERY_DIM:-}"
BEV_MEMORY_MAX_TOKENS="${BEV_MEMORY_MAX_TOKENS:-0}"

mkdir -p "$OUT_DIR"
cd "$NEW_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPU python eval_999.py \
    --sft_dir "$CKPT_DIR" \
    --output_dir "$OUT_DIR" \
    --data_path "$DATA_ROOT/data/3dtesting_val_999.json" \
    --voxelnext_token_mode "$VOXELNEXT_TOKEN_MODE" \
    --bev_query_num "$BEV_QUERY_NUM" \
    --bev_query_layers "$BEV_QUERY_LAYERS" \
    --bev_query_heads "$BEV_QUERY_HEADS" \
    ${BEV_QUERY_DIM:+--bev_query_dim "$BEV_QUERY_DIM"} \
    --bev_query_use_view_embed True \
    --bev_memory_max_tokens "$BEV_MEMORY_MAX_TOKENS" \
    "$@"

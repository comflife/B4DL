#!/bin/bash
# Generic eval on 3DTesting val set.
set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_dir> [output_dir] [gpu_id]" >&2
    exit 1
fi

CKPT_DIR="$1"
OUT_DIR="${2:-$CKPT_DIR/eval_val}"
GPU="${3:-0}"

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate "$QWEN_MM_ENV"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/0526040099_A/3davs_b4dl}"

mkdir -p "$OUT_DIR"
cd "$NEW_CODE_DIR"

CUDA_VISIBLE_DEVICES=$GPU python eval_q3d.py \
    --model_path "$CKPT_DIR" \
    --output_dir "$OUT_DIR" \
    --data_path "$DATA_ROOT/data/3dtesting_val_999.json" \
    --batch_size 1 \
    --num_workers 4 \
    "$@"

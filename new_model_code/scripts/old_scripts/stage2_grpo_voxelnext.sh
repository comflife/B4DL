#!/bin/bash
# Stage 2 — GRPO LoRA on top of the VoxelNeXt SFT-1b checkpoint.
#
# Same hyperparameters as the SMAP "safer" run (lr 5e-6, kl_coef 0.20,
# K=4 rollouts, max_new_tokens 128). The only changes are pointing at the
# VoxelNeXt SFT directory and the feature folder.
set -e

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate "$QWEN_MM_ENV"

# pcdet CUDA exts need libtorch on LD_LIBRARY_PATH and a matching nvcc.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${GPU:-0}"

cd "$NEW_CODE_DIR"
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=$GPU python -u train_qwen_grpo.py \
    --base_model Qwen/Qwen3.5-9B \
    --sft_dir "$DATA_ROOT/checkpoints/qwen_stage1b_vxnxt" \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version v1.0-trainval \
    --n_sweeps 10 \
    --voxelnext_root "$VOXELNEXT_ROOT" \
    --voxelnext_ckpt "$VOXELNEXT_CKPT" \
    --voxelnext_top_k 256 \
    --data_path "$DATA_ROOT/data/3dtesting_train_q3d.json" \
    --output_dir "$DATA_ROOT/checkpoints/qwen_stage2_grpo_vxnxt" \
    --sample_ratio 0.25 \
    --num_rollouts 4 \
    --gradient_accumulation_steps 4 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --top_p 0.9 \
    --learning_rate 5e-6 \
    --kl_coef 0.20 \
    --epochs 1 \
    --lora_r 32 \
    --lora_alpha 64 \
    --log_every 10 \
    --save_every 50 \
    "$@"

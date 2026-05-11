#!/bin/bash
# Stage 1b — Full FT with joint frozen VoxelNeXt encoding.
#
# Loads stage-1a warmed-up projector pair and continues full-parameter
# fine-tuning of the LLM (with the encoder still frozen). Raw 10-sweep
# nuScenes LiDAR is encoded live each step.
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
OUT_DIR="$DATA_ROOT/checkpoints/qwen_stage1b_vxnxt"
DATA_PATH="$DATA_ROOT/data/stage1_combined_q3d.json"
BASE_MODEL="Qwen/Qwen3.5-9B"
PRETRAIN_PROJ="$DATA_ROOT/checkpoints/qwen_stage1a_vxnxt/mm_projector.bin"

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
    --voxelnext_root "$VOXELNEXT_ROOT" \
    --voxelnext_ckpt "$VOXELNEXT_CKPT" \
    --voxelnext_top_k 256 \
    --voxelnext_freeze True \
    --pretrain_mm_projector "$PRETRAIN_PROJ" \
    --tune_mm_only False \
    --coord_aux_weight 0.5 \
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
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 1e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name qwen_stage1b_vxnxt \
    "$@"

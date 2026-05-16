#!/bin/bash
# Stage 1c — det_area-only visual-grounding refinement from stage2.
set -e

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate "$QWEN_MM_ENV"

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="$DATA_ROOT/checkpoints/qwen_lidarllm_stage3_detarea_999"
DATA_PATH="$DATA_ROOT/data/stage1c_detarea_999.json"
BASE_MODEL="$DATA_ROOT/checkpoints/qwen_lidarllm_stage2_ground_999"
VOXELNEXT_TOKEN_MODE="${VOXELNEXT_TOKEN_MODE:-bev_query}"
BEV_QUERY_NUM="${BEV_QUERY_NUM:-576}"
BEV_QUERY_LAYERS="${BEV_QUERY_LAYERS:-2}"
BEV_QUERY_HEADS="${BEV_QUERY_HEADS:-8}"
BEV_QUERY_DIM="${BEV_QUERY_DIM:-768}"
BEV_MEMORY_MAX_TOKENS="${BEV_MEMORY_MAX_TOKENS:-0}"
STAGE1C_EPOCHS="${STAGE1C_EPOCHS:-3}"
STAGE1C_LR="${STAGE1C_LR:-1e-5}"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT="${MASTER_PORT:-29585}"

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

if [ ! -d "$BASE_MODEL" ]; then
    echo "[error] missing stage1b checkpoint at $BASE_MODEL — run stage1b first" >&2
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
    --mm_pos_dim 0 \
    --voxelnext_root "$VOXELNEXT_ROOT" \
    --voxelnext_ckpt "$VOXELNEXT_CKPT" \
    --voxelnext_top_k 256 \
    --voxelnext_token_mode "$VOXELNEXT_TOKEN_MODE" \
    --bev_query_num "$BEV_QUERY_NUM" \
    --bev_query_layers "$BEV_QUERY_LAYERS" \
    --bev_query_heads "$BEV_QUERY_HEADS" \
    --bev_query_dim "$BEV_QUERY_DIM" \
    --bev_query_use_view_embed True \
    --bev_memory_max_tokens "$BEV_MEMORY_MAX_TOKENS" \
    --voxelnext_freeze True \
    --tune_mm_only False \
    --data_path "$DATA_PATH" \
    --template_filter det_area \
    --nuscenes_root "$NUSCENES_ROOT" \
    --nuscenes_version v1.0-trainval \
    --n_sweeps 10 \
    --max_length 2048 \
    --output_dir "$OUT_DIR" \
    --bf16 True \
    --num_train_epochs "$STAGE1C_EPOCHS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "no" \
    --remove_unused_columns False \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --learning_rate "$STAGE1C_LR" \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --report_to wandb \
    --run_name qwen_lidarllm_stage3_detarea_999 \
    "$@"

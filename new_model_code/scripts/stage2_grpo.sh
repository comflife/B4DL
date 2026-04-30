#!/bin/bash
# Stage 2 — Custom GRPO LoRA fine-tune on a 25 % sample of nuGrounding,
# using the LiDAR-aware composite reward (BEV-IoU + center + heading
# + safety + miss + class match for det_object).
set -e

source /home/byounggun/anaconda3/etc/profile.d/conda.sh
conda activate /data1/byounggun/conda_envs/qwen_mm

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPU="${GPU:-0}"

cd "$NEW_CODE_DIR"
export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES=$GPU python -u train_qwen_grpo.py \
    --base_model Qwen/Qwen3-0.6B \
    --sft_dir /data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1b_q3d \
    --feat_folder /data1/byounggun/3davs_b4dl/features/smap_lidar12 \
    --data_path /data1/byounggun/3davs_b4dl/data/3dtesting_train_q3d.json \
    --output_dir /data1/byounggun/3davs_b4dl/checkpoints/qwen_stage2_grpo_q3d_safe \
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

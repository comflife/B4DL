#!/usr/bin/env bash
# Single-shot B4DL launcher. Drop into a fresh tmux pane and run:
#
#     bash /home/hanyan_arch/B4DL/new_model_code/scripts/run_b4dl.sh
#
# Does, in order:
#   1. source conda + ~/.bashrc.b4dl, activate $QWEN_MM_ENV
#   2. set CUDA_HOME / LD_LIBRARY_PATH so pcdet's CUDA exts load
#   3. wandb env vars (project / run names)
#   4. chain: stage 1a -> 1b -> eval(post-1b) -> stage 2 -> eval(post-2)
#      logging to $DATA_ROOT/logs/run_all_<timestamp>.log via tee.
#
# Knobs (override on the command line, e.g. `GPUS=0,1 bash run_b4dl.sh`):
#   GPUS                default "0,1,2,3"
#   EVAL_MAX_SAMPLES    default 1000      (val slice for both eval steps)
#   WANDB_PROJECT       default "b4dl-vxnxt"
#   SKIP_STAGE1A / SKIP_STAGE1B / SKIP_EVAL / SKIP_STAGE2 / SKIP_EVAL_2
#                       set to 1 to skip individual phases

set -euo pipefail

# ---------- 1) shell + conda ----------------------------------------------
[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"

if [ -z "${CONDA_DEFAULT_ENV:-}" ] || [ "${CONDA_DEFAULT_ENV}" != "$(basename "$QWEN_MM_ENV")" ]; then
    source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
    conda activate "$QWEN_MM_ENV"
fi

# ---------- 2) CUDA + libtorch on the search paths ------------------------
# pcdet (VoxelNeXt) ext was built against the CUDA 12.8 nvcc + libtorch
# inside this conda env; both have to be visible at runtime too.
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0+PTX}"   # B200

# ---------- 3) wandb ------------------------------------------------------
export WANDB_PROJECT="${WANDB_PROJECT:-b4dl-vxnxt}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen_stage2_grpo_vxnxt}"
# stage 1a/1b TrainingArguments already pass --report_to wandb with their
# own run_name; this WANDB_PROJECT covers all three stages under one project.

if ! python -c "import wandb; wandb.api.api_key" >/dev/null 2>&1; then
    echo "[run_b4dl] !!! wandb not authenticated -- run 'wandb login' first or"
    echo "[run_b4dl]     export WANDB_API_KEY=...  (skipping wandb logging)"
    export WANDB_DISABLED=true
fi

# ---------- 4) chain ------------------------------------------------------
GPUS="${GPUS:-0,1,2,3}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
mkdir -p "$DATA_ROOT/logs"
LOGFILE="$DATA_ROOT/logs/run_all_$(date +%Y%m%d_%H%M%S).log"

echo "==========================================================="
echo " B4DL pipeline -- $(date)"
echo " GPUS=$GPUS  EVAL_MAX_SAMPLES=$EVAL_MAX_SAMPLES"
echo " WANDB_PROJECT=$WANDB_PROJECT (disabled=${WANDB_DISABLED:-no})"
echo " log: $LOGFILE"
echo "==========================================================="

GPUS="$GPUS" \
EVAL_MAX_SAMPLES="$EVAL_MAX_SAMPLES" \
WANDB_PROJECT="$WANDB_PROJECT" \
WANDB_RUN_NAME="$WANDB_RUN_NAME" \
SKIP_STAGE1A="${SKIP_STAGE1A:-0}" \
SKIP_STAGE1B="${SKIP_STAGE1B:-0}" \
SKIP_EVAL="${SKIP_EVAL:-0}" \
SKIP_STAGE2="${SKIP_STAGE2:-0}" \
SKIP_EVAL_2="${SKIP_EVAL_2:-0}" \
    bash "$REPO_ROOT/new_model_code/scripts/run_all_voxelnext.sh" 2>&1 | tee "$LOGFILE"

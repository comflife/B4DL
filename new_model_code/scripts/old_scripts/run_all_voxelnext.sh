#!/bin/bash
# Run the full VoxelNeXt training chain end-to-end:
#   Stage 1a (projector warmup, multi-GPU DDP)
#   Stage 1b (full FT, multi-GPU DDP)
#   Eval after 1b: LiDAR-LLM-comparable BEV mIoU on 3dtesting val
#   Stage 2 (GRPO LoRA, single GPU - the trl GRPO loop is single-process)
#
# Each stage's output is the next stage's input, so the chain stops as
# soon as anything fails (set -e). Logs go to per-stage files under each
# checkpoint directory; tail any of them to follow progress.
#
# Usage:
#   GPUS=0,1,2,3 bash scripts/run_all_voxelnext.sh
#   GPUS=0,1,2,3 SKIP_STAGE1A=1 bash scripts/run_all_voxelnext.sh   # resume from 1b
#   GPUS=0,1,2,3 SKIP_STAGE1A=1 SKIP_STAGE1B=1 bash scripts/run_all_voxelnext.sh
#
# Env knobs:
#   GPUS               default "0,1,2,3" (comma-separated CUDA_VISIBLE_DEVICES)
#   MASTER_PORT_1A     default 29583
#   MASTER_PORT_1B     default 29584
#   EVAL_MAX_SAMPLES   default 1000     (val slice for between-stage eval)
#   SKIP_STAGE1A       set 1 to skip    (requires existing mm_projector.bin)
#   SKIP_STAGE1B       set 1 to skip    (requires existing stage 1b ckpt)
#   SKIP_EVAL          set 1 to skip the post-1b eval
#   SKIP_STAGE2        set 1 to skip
set -e

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$NEW_CODE_DIR/scripts"

GPUS="${GPUS:-0,1,2,3}"
MASTER_PORT_1A="${MASTER_PORT_1A:-29583}"
MASTER_PORT_1B="${MASTER_PORT_1B:-29584}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"

[ -f "$HOME/.bashrc.b4dl" ] && source "$HOME/.bashrc.b4dl"
CKPT_ROOT="${DATA_ROOT:?DATA_ROOT not set; source ~/.bashrc.b4dl}/checkpoints"
STAGE1A_DIR="$CKPT_ROOT/qwen_stage1a_vxnxt"
STAGE1B_DIR="$CKPT_ROOT/qwen_stage1b_vxnxt"
STAGE2_DIR="$CKPT_ROOT/qwen_stage2_grpo_vxnxt"
EVAL_DIR="$CKPT_ROOT/qwen_stage1b_vxnxt/eval"

mkdir -p "$STAGE1A_DIR" "$STAGE1B_DIR" "$STAGE2_DIR" "$EVAL_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_stage() {
    local name="$1" log="$2"; shift 2
    echo "[$(stamp)] >>> $name  (log: $log)"
    "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ]; then
        echo "[$(stamp)] !!! $name failed with exit $rc — chain aborted" >&2
        exit $rc
    fi
    echo "[$(stamp)] <<< $name done"
}

# ---------- Stage 1a (projector warmup) ----------
if [ "${SKIP_STAGE1A:-0}" != "1" ]; then
    run_stage "stage1a (projector warmup)" "$STAGE1A_DIR/run_all.log" \
        env GPUS="$GPUS" MASTER_PORT="$MASTER_PORT_1A" \
        bash "$SCRIPTS/stage1a_voxelnext.sh"
else
    echo "[$(stamp)] skipping stage1a (SKIP_STAGE1A=1)"
fi

if [ ! -f "$STAGE1A_DIR/mm_projector.bin" ]; then
    echo "[error] expected $STAGE1A_DIR/mm_projector.bin after stage1a — abort" >&2
    exit 2
fi

# ---------- Stage 1b (full FT) ----------
if [ "${SKIP_STAGE1B:-0}" != "1" ]; then
    run_stage "stage1b (full FT)" "$STAGE1B_DIR/run_all.log" \
        env GPUS="$GPUS" MASTER_PORT="$MASTER_PORT_1B" \
        bash "$SCRIPTS/stage1b_voxelnext.sh"
else
    echo "[$(stamp)] skipping stage1b (SKIP_STAGE1B=1)"
fi

if [ ! -f "$STAGE1B_DIR/config.json" ]; then
    echo "[error] expected HF model files in $STAGE1B_DIR after stage1b — abort" >&2
    exit 2
fi

# ---------- Eval (LiDAR-LLM-comparable BEV mIoU) ----------
# Single-GPU eval against the held-out 3dtesting_val_q3d set. Reproduces
# the LiDAR-LLM Table 7 5-class mIoU breakdown (Car / Pedestrian / Bus /
# Truck / Construction_vehicle) plus ACC-10 / ACC-5 / Car-only mIoU. The
# slice size is bounded by EVAL_MAX_SAMPLES so we don't sit through 43k
# samples between stage 1b and stage 2.
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    EVAL_LOG="$EVAL_DIR/eval_${EVAL_MAX_SAMPLES}.log"
    EVAL_JSONL="$EVAL_DIR/eval_${EVAL_MAX_SAMPLES}.jsonl"
    FIRST_GPU="${GPUS%%,*}"
    run_stage "eval (LiDAR-LLM mIoU, GPU=$FIRST_GPU, n=$EVAL_MAX_SAMPLES)" "$EVAL_LOG" \
        env CUDA_VISIBLE_DEVICES="$FIRST_GPU" \
        bash -c "
[ -f \"\$HOME/.bashrc.b4dl\" ] && source \"\$HOME/.bashrc.b4dl\"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate \"\$QWEN_MM_ENV\"
export CUDA_HOME=\"\$CONDA_PREFIX\"
export PATH=\"\$CUDA_HOME/bin:\$PATH\"
export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib:\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}\"
export TORCH_CUDA_ARCH_LIST=\"\${TORCH_CUDA_ARCH_LIST:-10.0+PTX}\"
cd \"$NEW_CODE_DIR\"
python -u eval_q3d.py \\
    --sft_dir \"$STAGE1B_DIR\" \\
    --data_path \"\$DATA_ROOT/data/3dtesting_val_q3d.json\" \\
    --max_samples $EVAL_MAX_SAMPLES \\
    --max_new_tokens 80 \\
    --out_jsonl \"$EVAL_JSONL\"
"
else
    echo "[$(stamp)] skipping post-1b eval (SKIP_EVAL=1)"
fi

# ---------- Stage 2 (GRPO LoRA) ----------
# stage2 script uses a single GPU via $GPU; pick the first one from $GPUS.
# trl 0.17 GRPOTrainer's loop is single-process; multi-GPU here would need
# a vllm rollout server that the auto-mode env doesn't have set up yet.
if [ "${SKIP_STAGE2:-0}" != "1" ]; then
    FIRST_GPU="${GPUS%%,*}"
    run_stage "stage2 (GRPO LoRA, GPU=$FIRST_GPU)" "$STAGE2_DIR/run_all.log" \
        env GPU="$FIRST_GPU" \
        bash "$SCRIPTS/stage2_grpo_voxelnext.sh"
else
    echo "[$(stamp)] skipping stage2 (SKIP_STAGE2=1)"
fi

# ---------- Eval after Stage 2 (SFT + GRPO LoRA) ----------
# Pick the most-trained adapter dir (final/ if it exists, else last step_*).
if [ "${SKIP_EVAL_2:-0}" != "1" ]; then
    if [ -d "$STAGE2_DIR/final" ]; then
        ADAPTER="$STAGE2_DIR/final"
    else
        ADAPTER="$(ls -1d "$STAGE2_DIR"/step_* 2>/dev/null | sort -V | tail -1 || true)"
    fi
    if [ -z "$ADAPTER" ] || [ ! -d "$ADAPTER" ]; then
        echo "[$(stamp)] no GRPO adapter found under $STAGE2_DIR — skipping post-stage2 eval"
    else
        EVAL2_DIR="$STAGE2_DIR/eval"
        mkdir -p "$EVAL2_DIR"
        EVAL2_LOG="$EVAL2_DIR/eval_${EVAL_MAX_SAMPLES}.log"
        EVAL2_JSONL="$EVAL2_DIR/eval_${EVAL_MAX_SAMPLES}.jsonl"
        FIRST_GPU="${GPUS%%,*}"
        run_stage "eval-after-stage2 (SFT+GRPO, GPU=$FIRST_GPU, adapter=$ADAPTER)" "$EVAL2_LOG" \
            env CUDA_VISIBLE_DEVICES="$FIRST_GPU" \
            bash -c "
[ -f \"\$HOME/.bashrc.b4dl\" ] && source \"\$HOME/.bashrc.b4dl\"
source /home/hanyan_arch/miniconda3/etc/profile.d/conda.sh
conda activate \"\$QWEN_MM_ENV\"
export CUDA_HOME=\"\$CONDA_PREFIX\"
export PATH=\"\$CUDA_HOME/bin:\$PATH\"
export LD_LIBRARY_PATH=\"\$CUDA_HOME/lib:\$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:\${LD_LIBRARY_PATH:-}\"
export TORCH_CUDA_ARCH_LIST=\"\${TORCH_CUDA_ARCH_LIST:-10.0+PTX}\"
cd \"$NEW_CODE_DIR\"
python -u eval_q3d.py \\
    --sft_dir \"$STAGE1B_DIR\" \\
    --lora_adapter \"$ADAPTER\" \\
    --data_path \"\$DATA_ROOT/data/3dtesting_val_q3d.json\" \\
    --max_samples $EVAL_MAX_SAMPLES \\
    --max_new_tokens 80 \\
    --out_jsonl \"$EVAL2_JSONL\"
"
    fi
else
    echo "[$(stamp)] skipping post-stage2 eval (SKIP_EVAL_2=1)"
fi

echo "[$(stamp)] === all stages finished ==="
echo "  stage1a   -> $STAGE1A_DIR"
echo "  stage1b   -> $STAGE1B_DIR"
echo "  eval(1b)  -> $EVAL_DIR"
echo "  stage2    -> $STAGE2_DIR"
echo "  eval(1b+grpo) -> $STAGE2_DIR/eval"

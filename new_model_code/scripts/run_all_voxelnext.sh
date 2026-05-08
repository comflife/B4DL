#!/bin/bash
# Run the full VoxelNeXt training chain end-to-end:
#   Stage 1a (projector warmup) -> Stage 1b (full FT) -> Stage 2 (GRPO LoRA)
#
# Each stage's output is the next stage's input, so the chain stops as
# soon as anything fails (set -e). Logs go to per-stage files under each
# checkpoint directory; tail any of them to follow progress.
#
# Usage:
#   GPUS=2,3 bash scripts/run_all_voxelnext.sh
#   GPUS=2,3 SKIP_STAGE1A=1 bash scripts/run_all_voxelnext.sh   # resume from 1b
#   GPUS=2,3 SKIP_STAGE1A=1 SKIP_STAGE1B=1 bash scripts/run_all_voxelnext.sh
#
# Env knobs:
#   GPUS               default "0,1"   (comma-separated CUDA_VISIBLE_DEVICES)
#   MASTER_PORT_1A     default 29583
#   MASTER_PORT_1B     default 29584
#   SKIP_STAGE1A       set 1 to skip   (requires existing mm_projector.bin)
#   SKIP_STAGE1B       set 1 to skip   (requires existing stage 1b ckpt)
#   SKIP_STAGE2        set 1 to skip
set -e

NEW_CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS="$NEW_CODE_DIR/scripts"

GPUS="${GPUS:-0,1}"
MASTER_PORT_1A="${MASTER_PORT_1A:-29583}"
MASTER_PORT_1B="${MASTER_PORT_1B:-29584}"

CKPT_ROOT="/data1/byounggun/3davs_b4dl/checkpoints"
STAGE1A_DIR="$CKPT_ROOT/qwen_stage1a_vxnxt"
STAGE1B_DIR="$CKPT_ROOT/qwen_stage1b_vxnxt"
STAGE2_DIR="$CKPT_ROOT/qwen_stage2_grpo_vxnxt"

mkdir -p "$STAGE1A_DIR" "$STAGE1B_DIR" "$STAGE2_DIR"

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

# ---------- Stage 2 (GRPO LoRA) ----------
# stage2 script uses a single GPU via $GPU; pick the first one from $GPUS.
if [ "${SKIP_STAGE2:-0}" != "1" ]; then
    FIRST_GPU="${GPUS%%,*}"
    run_stage "stage2 (GRPO LoRA, GPU=$FIRST_GPU)" "$STAGE2_DIR/run_all.log" \
        env GPU="$FIRST_GPU" \
        bash "$SCRIPTS/stage2_grpo_voxelnext.sh"
else
    echo "[$(stamp)] skipping stage2 (SKIP_STAGE2=1)"
fi

echo "[$(stamp)] === all stages finished ==="
echo "  stage1a -> $STAGE1A_DIR"
echo "  stage1b -> $STAGE1B_DIR"
echo "  stage2  -> $STAGE2_DIR"

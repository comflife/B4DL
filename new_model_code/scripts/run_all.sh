#!/bin/bash
# Master pipeline — Stage1a → 1b → 1c → evals.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-/NHNHOME/WORKSPACE/0526040099_A/3davs_b4dl}"
CHECKPOINT_ROOT="$DATA_ROOT/checkpoints"

# Override per-run if desired.
GPUS="${GPUS:-0,1}"

STAGE1A_DIR="$CHECKPOINT_ROOT/qwen_stage1a_999"
STAGE1B_DIR="$CHECKPOINT_ROOT/qwen_stage1b_999"
STAGE1C_DIR="$CHECKPOINT_ROOT/qwen_stage1c_999"

echo "=============================================="
echo "VoxelNeXt+999-bin full training pipeline"
echo "GPUS=$GPUS | DATA_ROOT=$DATA_ROOT"
echo "=============================================="

# -------- Stage 1a --------
if [ ! -f "$STAGE1A_DIR/mm_projector.bin" ]; then
    echo "--- Stage 1a: projector warmup ---"
    bash "$SCRIPT_DIR/stage1a.sh"
else
    echo "--- Stage 1a: found projector at $STAGE1A_DIR, skip ---"
fi

# -------- Stage 1b --------
if [ ! -f "$STAGE1B_DIR/model.safetensors" ] && [ ! -f "$STAGE1B_DIR/pytorch_model.bin" ]; then
    echo "--- Stage 1b: full SFT, frozen VoxelNeXt ---"
    bash "$SCRIPT_DIR/stage1b.sh"
else
    echo "--- Stage 1b: checkpoint found at $STAGE1B_DIR, skip ---"
fi

# -------- Eval after 1b --------
echo "--- Eval stage 1b ---"
QUANT_MODE=999 GPUS=0 bash "$SCRIPT_DIR/eval.sh" "$STAGE1B_DIR" "$STAGE1B_DIR/eval_val" || true

# -------- Stage 1c --------
if [ ! -f "$STAGE1C_DIR/model.safetensors" ] && [ ! -f "$STAGE1C_DIR/pytorch_model.bin" ]; then
    echo "--- Stage 1c: det_area-only SFT, frozen VoxelNeXt ---"
    bash "$SCRIPT_DIR/stage1c.sh"
else
    echo "--- Stage 1c: checkpoint found at $STAGE1C_DIR, skip ---"
fi

# -------- Eval after 1c --------
echo "--- Eval stage 1c ---"
QUANT_MODE=999 GPUS=0 bash "$SCRIPT_DIR/eval.sh" "$STAGE1C_DIR" "$STAGE1C_DIR/eval_val" || true

echo "=============================================="
echo "Pipeline complete!"
echo "Stage 1a: $STAGE1A_DIR"
echo "Stage 1b: $STAGE1B_DIR"
echo "Stage 1c: $STAGE1C_DIR"
echo "=============================================="

#!/bin/bash
# Wait for SMAP extraction to finish, pick the two least-busy GPUs, and run
# Qwen3-0.6B stage 1a (projector warmup) -> stage 1b (full FT) in sequence.
set -u

FEAT_DIR="/data1/byounggun/3davs_b4dl/features/smap_lidar12"
TOK_LIST="/home/byounggun/B4DL/new_model_code/sample_tokens_union.json"
LOG_DIR="/data1/byounggun/3davs_b4dl/logs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WARMUP_SH="$SCRIPT_DIR/stage1a_warmup.sh"
FULL_SH="$SCRIPT_DIR/stage1b_full.sh"
WARMUP_PROJ="/data1/byounggun/3davs_b4dl/checkpoints/qwen_stage1a/mm_projector.bin"

mkdir -p "$LOG_DIR"

N_EXPECTED=$(python3 -c "import json;print(len(json.load(open('$TOK_LIST'))))")
N_REQUIRED=$(( N_EXPECTED * 99 / 100 ))

echo "[auto] waiting for >= $N_REQUIRED of $N_EXPECTED feature files..."
while :; do
    N_HAVE=$(ls "$FEAT_DIR" 2>/dev/null | wc -l)
    if [ "$N_HAVE" -ge "$N_REQUIRED" ]; then
        break
    fi
    echo "[auto] $(date +%T) progress=$N_HAVE/$N_EXPECTED"
    sleep 120
done
echo "[auto] extraction reached $N_HAVE/$N_EXPECTED — picking GPUs."

PICK=$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
                 --format=csv,noheader,nounits \
       | awk -F',' '{gsub(/ /,""); print $1, $2+0, $3+0}' \
       | sort -k2,2n -k3,3n \
       | awk '$2 < 4096 && $3 < 30 {print $1}' \
       | head -n2 | paste -sd,)

if [ -z "$PICK" ] || [ "$(echo "$PICK" | tr ',' '\n' | wc -l)" -lt 2 ]; then
    echo "[auto] could not find 2 free GPUs; current state:"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
    exit 2
fi

echo "[auto] launching Qwen stage1a (projector warmup) on GPUs=$PICK"
GPUS="$PICK" bash "$WARMUP_SH" > "$LOG_DIR/qwen_stage1a.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
    echo "[auto] stage1a FAILED rc=$RC — see $LOG_DIR/qwen_stage1a.log"
    exit $RC
fi

if [ ! -f "$WARMUP_PROJ" ]; then
    echo "[auto] stage1a finished but projector missing at $WARMUP_PROJ"
    exit 3
fi
echo "[auto] stage1a done. Projector at $WARMUP_PROJ."

echo "[auto] launching Qwen stage1b (full FT) on GPUs=$PICK"
GPUS="$PICK" bash "$FULL_SH" > "$LOG_DIR/qwen_stage1b.log" 2>&1
RC=$?
echo "[auto] stage1b finished rc=$RC"
exit $RC

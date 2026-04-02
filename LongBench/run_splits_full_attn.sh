#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-mistral}"
PRED_PATH="${PRED_PATH:-full_attn}"
BS="${BS:-1}"
CUDA="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR="logs/${MODEL}_${PRED_PATH}"
mkdir -p "$LOG_DIR"

PIDS=()

for SPLIT in 1 2 3 4 5; do
    echo "[$(date)] Launching split ${SPLIT} on CUDA_VISIBLE_DEVICES=${CUDA} ..."
    CUDA_VISIBLE_DEVICES=$CUDA python pred_single.py \
        --model "$MODEL" \
        --pred_path "$PRED_PATH" \
        -bs "$BS" \
        -ds "$SPLIT" \
        --no_morph
        > "${LOG_DIR}/full_attn_split_${SPLIT}.log" 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> ${LOG_DIR}/full_attn_split_${SPLIT}.log"
done

echo ""
echo "All 5 splits launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[ERROR] PID $PID exited with non-zero status"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[DONE] ${FAILED}/5 splits failed. Check logs in ${LOG_DIR}/"
    exit 1
else
    echo "[DONE] All 5 splits completed successfully."
fi

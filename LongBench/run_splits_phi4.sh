#!/bin/bash
set -euo pipefail

MODEL="${MODEL:-phi4}"
WS="${WS:-32}"
MC="${MC:-2048}"
IB="${IB:-0.8}"
FT="${FT:-2.0}"
PRED_PATH="${PRED_PATH:-hier_attn_cache}"
MORPH_TYPE="${MORPH_TYPE:-unimp_sum_fused}"
BS="${BS:-1}"
CUDA="${CUDA_VISIBLE_DEVICES:-2}"

LOG_DIR="logs/${MODEL}_${MORPH_TYPE}"
mkdir -p "$LOG_DIR"

PIDS=()

for SPLIT in 1 2 3; do
    echo "[$(date)] Launching split ${SPLIT} on CUDA_VISIBLE_DEVICES=${CUDA} ..."
    CUDA_VISIBLE_DEVICES=$CUDA python pred_single.py \
        --model "$MODEL" \
        --pred_path "$PRED_PATH" \
        --morph_type "$MORPH_TYPE" \
        -bs "$BS" \
        -ds "$SPLIT" \
        -ws "$WS" \
        -mc "$MC" \
        -ft "$FT" \
        -ib "$IB" \
        > "${LOG_DIR}/mkv_split_${SPLIT}.log" 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> ${LOG_DIR}/mkv_split_${SPLIT}.log"
done

echo ""
echo "First 3 splits launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[ERROR] PID $PID exited with non-zero status"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[DONE] ${FAILED}/3 splits failed. Check logs in ${LOG_DIR}/"
    exit 1
else
    echo "[DONE] First 3 splits completed successfully."
fi

for SPLIT in 4 5; do
    echo "[$(date)] Launching split ${SPLIT} on CUDA_VISIBLE_DEVICES=${CUDA} ..."
    CUDA_VISIBLE_DEVICES=$CUDA python pred_single.py \
        --model "$MODEL" \
        --pred_path "$PRED_PATH" \
        --morph_type "$MORPH_TYPE" \
        -bs "$BS" \
        -ds "$SPLIT" \
        -ws "$WS" \
        -mc "$MC" \
        -ft "$FT" \
        -ib "$IB" \
        > "${LOG_DIR}/mkv_split_${SPLIT}.log" 2>&1 &
    PIDS+=($!)
    echo "  PID=$! -> ${LOG_DIR}/mkv_split_${SPLIT}.log"
done

echo ""
echo "Last 2 splits launched. PIDs: ${PIDS[*]}"
echo "Waiting for all to finish..."

FAILED=0
for PID in "${PIDS[@]}"; do
    if ! wait "$PID"; then
        echo "[ERROR] PID $PID exited with non-zero status"
        FAILED=$((FAILED + 1))
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo "[DONE] ${FAILED}/2 splits failed. Check logs in ${LOG_DIR}/"
    exit 1
else
    echo "[DONE] Last 2 splits completed successfully."
fi

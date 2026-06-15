#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Evaluate a merged Res-OPD checkpoint using CHAIR metrics
#
# Steps:
#   1. Start vLLM server with the merged checkpoint
#   2. Run CHAIR evaluation via API
#   3. Shut down vLLM server
#
# Usage:
#   bash scripts/eval_after_merge.sh <merged_checkpoint_path> [student_px]
#
# Eval results are saved to res-opd/eval_results/<experiment_name>/<dataset_tag>/.
# dataset_tag is auto-derived from data files, e.g. train1500_test300.
#
# Example:
#   bash scripts/eval_after_merge.sh \
#       ./checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s224-a0.5/global_step_50/ \
#       224
#   # Results → ./eval_results/Res-OPD-Qwen3VL-2B-Instruct-s224-a0.5/train1500_test300/
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [student_px]}"
STUDENT_PX="${2:-0}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="Res-OPD"
TEST_JSON="${RES_OPD_ROOT}/data/test.json"

# Determine eval output directory:
#   Always store under res-opd/eval_results/<experiment_name>/<dataset_tag>/
#   dataset_tag is derived from train/test file sizes, e.g. train1500_test300
CKPT_ROOT="$MODEL_PATH"
if [[ "$(basename "$MODEL_PATH")" =~ ^global_step_[0-9]+$ ]]; then
    CKPT_ROOT="$(dirname "$MODEL_PATH")"
fi
EXPERIMENT_NAME="$(basename "$CKPT_ROOT")"

# Build dataset tag from actual data file sizes
TRAIN_FILE="${RES_OPD_ROOT}/data/train.parquet"
TEST_FILE="${RES_OPD_ROOT}/data/test.json"
DATASET_TAG=""
if [[ -f "$TRAIN_FILE" && -f "$TEST_FILE" ]]; then
    TRAIN_N=$(/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))" 2>/dev/null || echo "?")
    TEST_N=$(/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -c "import json; print(len(json.load(open('$TEST_FILE'))))" 2>/dev/null || echo "?")
    DATASET_TAG="train${TRAIN_N}_test${TEST_N}"
else
    DATASET_TAG="unknown_dataset"
fi
OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${EXPERIMENT_NAME}/${DATASET_TAG}"

if [ ! -f "$TEST_JSON" ]; then
    echo "Error: test.json not found at $TEST_JSON" >&2
    echo "Run prepare_data.py first." >&2
    exit 1
fi

echo "============================================================"
echo " Res-OPD CHAIR Evaluation"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "Student px:  $STUDENT_PX (0 = original image)"
echo "Test data:   $TEST_JSON"
echo "Output:      $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# --- Step 1: Start vLLM server ---
echo ""
echo "[1/3] Starting vLLM server on port $PORT ..."
/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization 0.85 \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --port "$PORT" \
    --max-model-len 9728 &
VLLM_PID=$!

# Wait for server to be ready
echo "  Waiting for vLLM server (pid=$VLLM_PID) ..."
for i in $(seq 1 300); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  vLLM server exited unexpectedly!" >&2
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  vLLM server failed to start within 300s" >&2
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# --- Step 2: Run evaluation ---
echo ""
echo "[2/3] Running CHAIR evaluation ..."
/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 "${RES_OPD_ROOT}/eval/eval_chair.py" \
    --api-base "http://localhost:$PORT/v1/" \
    --model-name "$MODEL_NAME" \
    --test-json "$TEST_JSON" \
    --output-dir "$OUTPUT_DIR" \
    --student-px "$STUDENT_PX"

# --- Step 3: Cleanup ---
echo ""
echo "[3/3] Shutting down vLLM server ..."
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_DIR"

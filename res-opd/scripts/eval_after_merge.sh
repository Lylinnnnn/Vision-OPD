#!/usr/bin/env bash

set -euo pipefail

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ⚠️  IMPORTANT: 一次性/临时 eval 脚本（如批量eval、对比实验等）必须放在
#     scripts/tmp/ 目录下，禁止直接放在 scripts/ 根目录！
#     scripts/ 只保留常规最小运行脚本。用完的临时脚本请及时删除或归档。
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Evaluate a merged Res-OPD checkpoint using CHAIR metrics
#
# Steps:
#   1. Start vLLM server with the merged checkpoint
#   2. Run CHAIR evaluation via API
#   3. Shut down vLLM server
#
# Usage:
#   bash scripts/eval_after_merge.sh <merged_checkpoint_path> [student_px] [version_tag]
#
# Output directory structure (unified naming):
#   res-opd/eval_results/<version_tag>/<experiment_name>_<step_tag>/<dataset_tag>/
#
#   - version_tag:  e.g. "v5", "v4_frozen", "debug" (default: "latest")
#   - experiment_name: derived from checkpoint parent dir name
#   - step_tag: e.g. "global_step_46" (auto-detected from path)
#   - dataset_tag: auto-derived from data files, e.g. "train1500_test300"
#
# Examples:
#   # Default version tag "latest"
#   bash scripts/eval_after_merge.sh \
#       ./checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s448-t200-a0.5-ema-e2/global_step_92/ \
#       448
#   # → eval_results/latest/Res-OPD-...-s448-t200-..._global_step_92/train1500_test300/
#
#   # Custom version tag
#   bash scripts/eval_after_merge.sh \
#       ./checkpoints/Res-OPD-.../global_step_92/ 448 v5
#   # → eval_results/v5/Res-OPD-..._global_step_92/train1500_test300/
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [student_px] [version_tag]}"
STUDENT_PX="${2:-0}"
VERSION_TAG="${3:-latest}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="Res-OPD"
TEST_JSON="${RES_OPD_ROOT}/data/test.json"
PYTHON_BIN="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3"

# Determine eval output directory:
#   res-opd/eval_results/<version_tag>/<experiment_name>_<step_tag>/<dataset_tag>/
CKPT_ROOT="$MODEL_PATH"
STEP_TAG=""
if [[ "$(basename "$MODEL_PATH")" =~ ^global_step_[0-9]+$ ]]; then
    STEP_TAG="$(basename "$MODEL_PATH")"
    CKPT_ROOT="$(dirname "$MODEL_PATH")"
fi
EXPERIMENT_NAME="$(basename "$CKPT_ROOT")"
# Append step tag to distinguish different checkpoints of the same experiment
if [[ -n "$STEP_TAG" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${STEP_TAG}"
fi

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
OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${EXPERIMENT_NAME}/${DATASET_TAG}"

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
# Disable prometheus metrics to avoid '_IncludedRouter' compatibility issue with vLLM 0.18+
export VLLM_DISABLE_PROMETHEUS=1
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization 0.85 \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --port "$PORT" \
    --max-model-len 9728 \
    --disable-frontend-multiprocessing &
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
"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_chair.py" \
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

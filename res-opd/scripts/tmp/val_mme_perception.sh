#!/usr/bin/env bash

set -euo pipefail

# Final-only classic MME perception evaluation.
#
# Usage with official MME root:
#   MME_ROOT=/path/to/MME_Benchmark_release_version \
#   bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> [version_tag]
#
# Usage with converted JSON/JSONL:
#   MME_JSON=/path/to/mme_perception.json \
#   bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> [version_tag]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [version_tag]}"
VERSION_TAG="${2:-${VERSION_TAG:-latest}}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-Res-OPD}"
MME_CATEGORIES="${MME_CATEGORIES:-existence,count,position,color}"
MME_MAX_SAMPLES="${MME_MAX_SAMPLES:-0}"
MME_PARALLEL_WORKERS="${MME_PARALLEL_WORKERS:-64}"

CKPT_ROOT="$MODEL_PATH"
STEP_TAG=""
if [[ "$(basename "$MODEL_PATH")" =~ ^global_step_[0-9]+$ ]]; then
    STEP_TAG="$(basename "$MODEL_PATH")"
    CKPT_ROOT="$(dirname "$MODEL_PATH")"
fi
EXPERIMENT_NAME="$(basename "$CKPT_ROOT")"
[[ -n "$STEP_TAG" ]] && EXPERIMENT_NAME="${EXPERIMENT_NAME}_${STEP_TAG}"
OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${EXPERIMENT_NAME}/final_hallucination"

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$(cd "$RES_OPD_ROOT/.." && pwd):${PYTHONPATH:-}"
export VLLM_DISABLE_PROMETHEUS=1

echo "============================================================"
echo " Classic MME Perception Eval"
echo "============================================================"
echo "Model:      $MODEL_PATH"
echo "Categories: $MME_CATEGORIES"
echo "Output:     $OUTPUT_DIR"
echo "============================================================"

"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --port "$PORT" \
    --max-model-len "${VLLM_MAX_MODEL_LEN:-9728}" \
    --disable-frontend-multiprocessing &
VLLM_PID=$!

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

for i in $(seq 1 300); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "vLLM server exited unexpectedly." >&2
        exit 1
    fi
    sleep 1
done

source_args=()
if [[ -n "${MME_JSON:-}" ]]; then
    source_args+=(--mme-json "$MME_JSON")
elif [[ -n "${MME_ROOT:-}" ]]; then
    source_args+=(--mme-root "$MME_ROOT" --categories "$MME_CATEGORIES")
else
    echo "Error: set MME_ROOT for official MME data or MME_JSON for converted data." >&2
    exit 1
fi

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_mme_classic.py" \
    --api-base "http://localhost:$PORT/v1/" \
    --api-key "${OPENAI_API_KEY:-EMPTY}" \
    --model-name "$MODEL_NAME" \
    "${source_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --max-samples "$MME_MAX_SAMPLES" \
    --parallel-workers "$MME_PARALLEL_WORKERS"

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/collect_benchmark_summary.py" \
    --output-dir "$OUTPUT_DIR" || true

cleanup
trap - EXIT

echo "MME perception eval complete: $OUTPUT_DIR"

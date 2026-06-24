#!/usr/bin/env bash

set -euo pipefail

# Final-only AMBER hallucination evaluation.
#
# Usage:
#   AMBER_ROOT=/path/to/AMBER AMBER_IMAGE_ROOT=/path/to/AMBER/images \
#   bash res-opd/scripts/tmp/val_amber.sh <merged_checkpoint_path> [version_tag]
#
# Low-disk staging:
#   AMBER_OSS_URI=oss://bucket/path/AMBER \
#   bash res-opd/scripts/tmp/val_amber.sh <merged_checkpoint_path> [version_tag]
#
# Optional ModelScope fallback:
#   AMBER_MODELSCOPE_ID=<owner/dataset> \
#   bash res-opd/scripts/tmp/val_amber.sh <merged_checkpoint_path> [version_tag]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [version_tag]}"
VERSION_TAG="${2:-${VERSION_TAG:-latest}}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-Res-OPD}"
BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-/home/liuyanlin.lyl/notebook/data}"
BENCHMARK_OSS_BASE="${BENCHMARK_OSS_BASE:-}"
AMBER_ROOT="${AMBER_ROOT:-${BENCHMARK_DATA_ROOT}/AMBER}"
AMBER_OSS_URI="${AMBER_OSS_URI:-${BENCHMARK_OSS_BASE:+${BENCHMARK_OSS_BASE%/}/AMBER}}"
AMBER_MODELSCOPE_ID="${AMBER_MODELSCOPE_ID:-}"
KEEP_BENCHMARK_DATA="${KEEP_BENCHMARK_DATA:-False}"
AMBER_EVAL_TYPE="${AMBER_EVAL_TYPE:-a}"
AMBER_MAX_SAMPLES="${AMBER_MAX_SAMPLES:-0}"
AMBER_PARALLEL_WORKERS="${AMBER_PARALLEL_WORKERS:-64}"
STUDENT_PX="${STUDENT_PX:-}"
TARGET_PX="${TARGET_PX:-448}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
STAGED_DATASET=0

CKPT_ROOT="$MODEL_PATH"
STEP_TAG=""
if [[ "$(basename "$MODEL_PATH")" =~ ^global_step_[0-9]+$ ]]; then
    STEP_TAG="$(basename "$MODEL_PATH")"
    CKPT_ROOT="$(dirname "$MODEL_PATH")"
fi
EXPERIMENT_NAME="$(basename "$CKPT_ROOT")"
[[ -n "$STEP_TAG" ]] && EXPERIMENT_NAME="${EXPERIMENT_NAME}_${STEP_TAG}"
OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${EXPERIMENT_NAME}/final_hallucination"

infer_eval_spec() {
    local exp_name="$1"
    local inferred_mode="square"
    local inferred_student_px="0"
    local inferred_student_ratio="1.0"
    if [[ "$exp_name" =~ -orig-sr([0-9.]+)-tr ]]; then
        inferred_mode="original"
        inferred_student_px="0"
        inferred_student_ratio="${BASH_REMATCH[1]}"
    elif [[ "$exp_name" =~ -s([0-9]+)(-|_) ]]; then
        inferred_mode="square"
        inferred_student_px="${BASH_REMATCH[1]}"
    fi
    DEGRADATION_MODE="${DEGRADATION_MODE:-$inferred_mode}"
    STUDENT_PX="${STUDENT_PX:-$inferred_student_px}"
    STUDENT_RATIO="${STUDENT_RATIO:-$inferred_student_ratio}"
}

infer_eval_spec "$EXPERIMENT_NAME"

stage_from_oss() {
    local oss_uri="$1"
    local local_dir="$2"
    if [[ -z "$oss_uri" ]]; then
        return 1
    fi
    echo "Staging AMBER from OSS: ${oss_uri} -> ${local_dir}"
    mkdir -p "$local_dir"
    ossutil cp -r "${oss_uri%/}/" "$local_dir/" -f
}

stage_from_modelscope() {
    local dataset_id="$1"
    local local_dir="$2"
    if [[ -z "$dataset_id" ]]; then
        return 1
    fi
    if ! command -v modelscope >/dev/null 2>&1; then
        echo "ModelScope CLI not found; skipping ModelScope download." >&2
        return 1
    fi
    echo "Staging AMBER from ModelScope dataset: ${dataset_id} -> ${local_dir}"
    mkdir -p "$local_dir"
    modelscope download --dataset "$dataset_id" --local_dir "$local_dir"
}

amber_image_present() {
    if [[ -n "${AMBER_IMAGE_ROOT:-}" && -f "${AMBER_IMAGE_ROOT}/AMBER_1.jpg" ]]; then
        return 0
    fi
    local candidates=(
        "${AMBER_ROOT}/AMBER_1.jpg"
        "${AMBER_ROOT}/images/AMBER_1.jpg"
        "${AMBER_ROOT}/image/AMBER_1.jpg"
        "${AMBER_ROOT}/data/images/AMBER_1.jpg"
    )
    for candidate in "${candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            return 0
        fi
    done
    return 1
}

ensure_amber_data() {
    local query_file="${AMBER_ROOT}/data/query/query_all.json"
    if [[ -f "$query_file" ]] && amber_image_present; then
        return
    fi
    if [[ ! -f "$query_file" || ! amber_image_present ]]; then
        if stage_from_oss "$AMBER_OSS_URI" "$AMBER_ROOT"; then
            STAGED_DATASET=1
        elif stage_from_modelscope "$AMBER_MODELSCOPE_ID" "$AMBER_ROOT"; then
            STAGED_DATASET=1
        fi
    fi
    if [[ ! -f "$query_file" ]]; then
        echo "Error: AMBER metadata not found at ${AMBER_ROOT}/data." >&2
        echo "Set AMBER_ROOT, or set AMBER_OSS_URI, or set AMBER_MODELSCOPE_ID." >&2
        echo "After manual download, you can upload with:" >&2
        echo "  ossutil cp -r ${AMBER_ROOT}/ oss://<bucket>/<path>/AMBER/ -f" >&2
        cleanup_dataset
        exit 1
    fi
    if ! amber_image_present; then
        echo "Error: AMBER image AMBER_1.jpg not found." >&2
        echo "Expected one of:" >&2
        echo "  ${AMBER_ROOT}/AMBER_1.jpg" >&2
        echo "  ${AMBER_ROOT}/images/AMBER_1.jpg" >&2
        echo "  ${AMBER_ROOT}/image/AMBER_1.jpg" >&2
        echo "  ${AMBER_ROOT}/data/images/AMBER_1.jpg" >&2
        echo "Or set AMBER_IMAGE_ROOT to a directory containing AMBER_1.jpg." >&2
        cleanup_dataset
        exit 1
    fi
}

cleanup_dataset() {
    if [[ "$STAGED_DATASET" -eq 1 && "$KEEP_BENCHMARK_DATA" != "True" && "$KEEP_BENCHMARK_DATA" != "true" ]]; then
        echo "[cleanup] Removing staged AMBER data: $AMBER_ROOT"
        rm -rf "$AMBER_ROOT"
    fi
}

ensure_amber_data
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$(cd "$RES_OPD_ROOT/.." && pwd):${PYTHONPATH:-}"
export VLLM_DISABLE_PROMETHEUS=1

echo "============================================================"
echo " AMBER Eval"
echo "============================================================"
echo "Model:     $MODEL_PATH"
echo "AMBER:     $AMBER_ROOT"
echo "AMBER OSS: ${AMBER_OSS_URI:-<none>}"
echo "Eval type: $AMBER_EVAL_TYPE"
echo "Student:   mode=$DEGRADATION_MODE px=$STUDENT_PX target=$TARGET_PX ratio=$STUDENT_RATIO"
echo "Output:    $OUTPUT_DIR"
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
    cleanup_dataset
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

amber_args=()
[[ -n "${AMBER_IMAGE_ROOT:-}" ]] && amber_args+=(--image-root "$AMBER_IMAGE_ROOT")
[[ "${AMBER_SKIP_OFFICIAL_EVAL:-False}" == "True" || "${AMBER_SKIP_OFFICIAL_EVAL:-False}" == "true" ]] && amber_args+=(--skip-official-eval)

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_amber.py" \
    --api-base "http://localhost:$PORT/v1/" \
    --api-key "${OPENAI_API_KEY:-EMPTY}" \
    --model-name "$MODEL_NAME" \
    --amber-root "$AMBER_ROOT" \
    "${amber_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --student-px "$STUDENT_PX" \
    --target-px "$TARGET_PX" \
    --degradation-mode "$DEGRADATION_MODE" \
    --student-ratio "$STUDENT_RATIO" \
    --evaluation-type "$AMBER_EVAL_TYPE" \
    --max-samples "$AMBER_MAX_SAMPLES" \
    --parallel-workers "$AMBER_PARALLEL_WORKERS"

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/collect_benchmark_summary.py" \
    --output-dir "$OUTPUT_DIR" || true

cleanup
trap - EXIT

echo "AMBER eval complete: $OUTPUT_DIR"

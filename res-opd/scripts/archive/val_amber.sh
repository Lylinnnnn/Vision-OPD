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
DATASET_VERSION="${DATASET_VERSION:-full}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-Res-OPD}"
MODEL_PROFILE="${MODEL_PROFILE:-}"
BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-/home/liuyanlin.lyl/notebook/data}"
BENCHMARK_OSS_BASE="${BENCHMARK_OSS_BASE:-}"
AMBER_ROOT="${AMBER_ROOT:-${BENCHMARK_DATA_ROOT}/AMBER}"
AMBER_OSS_URI="${AMBER_OSS_URI:-${BENCHMARK_OSS_BASE:+${BENCHMARK_OSS_BASE%/}/AMBER}}"
AMBER_MODELSCOPE_ID="${AMBER_MODELSCOPE_ID:-}"
KEEP_BENCHMARK_DATA="${KEEP_BENCHMARK_DATA:-False}"
AMBER_EVAL_TYPE="${AMBER_EVAL_TYPE:-a}"
AMBER_MAX_SAMPLES="${AMBER_MAX_SAMPLES:-0}"
AMBER_PARALLEL_WORKERS_WAS_SET="${AMBER_PARALLEL_WORKERS+x}"
AMBER_PARALLEL_WORKERS="${AMBER_PARALLEL_WORKERS:-64}"
AMBER_OFFICIAL_EVAL_WORKERS="${AMBER_OFFICIAL_EVAL_WORKERS:-16}"
AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET="${AMBER_MAX_NEW_TOKENS_GENERATIVE+x}"
AMBER_MAX_NEW_TOKENS_GENERATIVE="${AMBER_MAX_NEW_TOKENS_GENERATIVE:-384}"
AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET="${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE+x}"
AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN_WAS_SET="${VLLM_MAX_MODEL_LEN+x}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-9728}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_MAX_NUM_SEQS_WAS_SET="${VLLM_MAX_NUM_SEQS+x}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
ENABLE_THINKING_WAS_SET=""
if [[ -n "${ENABLE_THINKING:-}" || -n "${VISION_ENABLE_THINKING:-}" ]]; then
    ENABLE_THINKING_WAS_SET="yes"
fi
ENABLE_THINKING="${ENABLE_THINKING:-${VISION_ENABLE_THINKING:-}}"
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

infer_eval_spec() {
    local exp_name="$1"
    local inferred_mode="original"
    local inferred_student_px="0"
    local inferred_student_ratio="1.0"
    if [[ "$exp_name" =~ -orig-sr([0-9.]+)-tr ]]; then
        inferred_student_px="0"
        inferred_student_ratio="${BASH_REMATCH[1]}"
    elif [[ "$exp_name" =~ -s([0-9]+)(-|_) ]]; then
        inferred_student_px="${BASH_REMATCH[1]}"
    fi
    DEGRADATION_MODE="${DEGRADATION_MODE:-$inferred_mode}"
    STUDENT_PX="${STUDENT_PX:-$inferred_student_px}"
    STUDENT_RATIO="${STUDENT_RATIO:-$inferred_student_ratio}"
}

infer_eval_spec "$EXPERIMENT_NAME"

if [[ "$DATASET_VERSION" == "full" ]]; then
    TRAIN_FILE="${RES_OPD_ROOT}/data/train_5k.parquet"
    TEST_FILE="${RES_OPD_ROOT}/data/test_1000.json"
else
    TRAIN_FILE="${RES_OPD_ROOT}/data/train.parquet"
    TEST_FILE="${RES_OPD_ROOT}/data/test.json"
fi
DATASET_TAG="unknown_dataset"
if [[ -f "$TRAIN_FILE" && -f "$TEST_FILE" ]]; then
    TRAIN_N=$("$PYTHON_BIN" -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))" 2>/dev/null || echo "?")
    TEST_N=$("$PYTHON_BIN" -c "import json; print(len(json.load(open('$TEST_FILE'))))" 2>/dev/null || echo "?")
    DATASET_TAG="train${TRAIN_N}_test${TEST_N}"
fi
if [[ "$DEGRADATION_MODE" == "original" ]]; then
    RATIO_TAG="${STUDENT_RATIO//./p}"
    DATASET_TAG="${DATASET_TAG}_original_sr${RATIO_TAG}"
fi
if [[ "$DATASET_VERSION" == "full" ]]; then
    OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/full/${EXPERIMENT_NAME}/${DATASET_TAG}"
else
    OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${EXPERIMENT_NAME}/${DATASET_TAG}"
fi

infer_model_profile() {
    local explicit_profile="$1"
    local path_lc
    if [[ -n "$explicit_profile" ]]; then
        echo "$explicit_profile"
        return 0
    fi
    path_lc="$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
    if [[ "$path_lc" == *"thinking"* ]]; then
        if [[ "$path_lc" == *"8b"* ]]; then
            echo "qwen3vl_8b_thinking"
        elif [[ "$path_lc" == *"2b"* ]]; then
            echo "qwen3vl_2b_thinking"
        else
            echo "generic_thinking"
        fi
    elif [[ "$path_lc" == *"instruct"* ]]; then
        echo "qwen3vl_instruct"
    else
        echo "default"
    fi
}

apply_model_profile_defaults() {
    local profile="$1"
    case "$profile" in
        qwen3vl_2b_thinking)
            if [[ -z "$ENABLE_THINKING_WAS_SET" ]]; then ENABLE_THINKING="True"; fi
            if [[ -z "$VLLM_MAX_MODEL_LEN_WAS_SET" ]]; then VLLM_MAX_MODEL_LEN="${THINKING_2B_VLLM_MAX_MODEL_LEN:-12288}"; fi
            if [[ -z "$VLLM_MAX_NUM_SEQS_WAS_SET" ]]; then VLLM_MAX_NUM_SEQS="${THINKING_2B_VLLM_MAX_NUM_SEQS:-32}"; fi
            if [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]]; then AMBER_MAX_NEW_TOKENS_GENERATIVE="${THINKING_2B_AMBER_MAX_NEW_TOKENS_GENERATIVE:-1152}"; fi
            if [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]]; then AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${THINKING_2B_AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-256}"; fi
            if [[ -z "$AMBER_PARALLEL_WORKERS_WAS_SET" ]]; then AMBER_PARALLEL_WORKERS="${THINKING_2B_AMBER_PARALLEL_WORKERS:-8}"; fi
            ;;
        qwen3vl_8b_thinking)
            if [[ -z "$ENABLE_THINKING_WAS_SET" ]]; then ENABLE_THINKING="True"; fi
            if [[ -z "$VLLM_MAX_MODEL_LEN_WAS_SET" ]]; then VLLM_MAX_MODEL_LEN="${THINKING_8B_VLLM_MAX_MODEL_LEN:-12288}"; fi
            if [[ -z "$VLLM_MAX_NUM_SEQS_WAS_SET" ]]; then VLLM_MAX_NUM_SEQS="${THINKING_8B_VLLM_MAX_NUM_SEQS:-8}"; fi
            if [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]]; then AMBER_MAX_NEW_TOKENS_GENERATIVE="${THINKING_8B_AMBER_MAX_NEW_TOKENS_GENERATIVE:-1152}"; fi
            if [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]]; then AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${THINKING_8B_AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-256}"; fi
            if [[ -z "$AMBER_PARALLEL_WORKERS_WAS_SET" ]]; then AMBER_PARALLEL_WORKERS="${THINKING_8B_AMBER_PARALLEL_WORKERS:-4}"; fi
            ;;
        generic_thinking)
            if [[ -z "$ENABLE_THINKING_WAS_SET" ]]; then ENABLE_THINKING="True"; fi
            if [[ -z "$VLLM_MAX_MODEL_LEN_WAS_SET" ]]; then VLLM_MAX_MODEL_LEN="${THINKING_VLLM_MAX_MODEL_LEN:-12288}"; fi
            if [[ -z "$VLLM_MAX_NUM_SEQS_WAS_SET" ]]; then VLLM_MAX_NUM_SEQS="${THINKING_VLLM_MAX_NUM_SEQS:-8}"; fi
            if [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]]; then AMBER_MAX_NEW_TOKENS_GENERATIVE="${THINKING_AMBER_MAX_NEW_TOKENS_GENERATIVE:-1152}"; fi
            if [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]]; then AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${THINKING_AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-256}"; fi
            if [[ -z "$AMBER_PARALLEL_WORKERS_WAS_SET" ]]; then AMBER_PARALLEL_WORKERS="${THINKING_AMBER_PARALLEL_WORKERS:-4}"; fi
            ;;
        qwen3vl_instruct|default)
            ;;
        *)
            echo "Error: unsupported MODEL_PROFILE=${profile}." >&2
            exit 1
            ;;
    esac
    return 0
}

MODEL_PROFILE="$(infer_model_profile "$MODEL_PROFILE")"
apply_model_profile_defaults "$MODEL_PROFILE"

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
echo "Dataset:   $DATASET_VERSION / $DATASET_TAG"
echo "Model profile: $MODEL_PROFILE"
echo "Enable thinking: ${ENABLE_THINKING:-<unset>}"
echo "AMBER max tokens: generative=${AMBER_MAX_NEW_TOKENS_GENERATIVE}, discriminative=${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE}"
echo "AMBER workers: $AMBER_PARALLEL_WORKERS"
echo "AMBER official eval workers: $AMBER_OFFICIAL_EVAL_WORKERS"
echo "Student:   mode=$DEGRADATION_MODE px=$STUDENT_PX target=$TARGET_PX ratio=$STUDENT_RATIO"
echo "vLLM:      max_len=$VLLM_MAX_MODEL_LEN max_num_seqs=${VLLM_MAX_NUM_SEQS:-<default>} tp=${VLLM_TENSOR_PARALLEL_SIZE:-<default>}"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"

vllm_args=(
    -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --port "$PORT"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --disable-frontend-multiprocessing
)
if [[ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]]; then
    vllm_args+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
fi
if [[ -n "$VLLM_MAX_NUM_SEQS" ]]; then
    vllm_args+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
"$PYTHON_BIN" "${vllm_args[@]}" &
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
[[ -n "$ENABLE_THINKING" ]] && amber_args+=(--enable-thinking "$ENABLE_THINKING")

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
    --max-new-tokens-generative "$AMBER_MAX_NEW_TOKENS_GENERATIVE" \
    --max-new-tokens-discriminative "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE" \
    --max-samples "$AMBER_MAX_SAMPLES" \
    --parallel-workers "$AMBER_PARALLEL_WORKERS" \
    --official-eval-workers "$AMBER_OFFICIAL_EVAL_WORKERS"

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/collect_benchmark_summary.py" \
    --output-dir "$OUTPUT_DIR" || true

cleanup
trap - EXIT

echo "AMBER eval complete: $OUTPUT_DIR"

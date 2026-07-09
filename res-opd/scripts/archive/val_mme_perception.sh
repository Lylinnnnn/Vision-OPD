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
#
# Usage with HuggingFace lmms-lab/MME parquet:
#   MME_HF_ROOT=/home/liuyanlin.lyl/notebook/data/MME_hf \
#   bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> [version_tag]
#
# Low-disk staging:
#   MME_HF_OSS_URI=oss://bucket/path/MME_hf \
#   bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> [version_tag]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [version_tag]}"
VERSION_TAG="${2:-${VERSION_TAG:-latest}}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
PORT="${VLLM_PORT:-8000}"
MODEL_NAME="${MODEL_NAME:-Res-OPD}"
BENCHMARK_DATA_ROOT="${BENCHMARK_DATA_ROOT:-/home/liuyanlin.lyl/notebook/data}"
BENCHMARK_OSS_BASE="${BENCHMARK_OSS_BASE:-}"
MME_ROOT="${MME_ROOT:-${BENCHMARK_DATA_ROOT}/MME_Benchmark_release_version}"
MME_OSS_URI="${MME_OSS_URI:-${BENCHMARK_OSS_BASE:+${BENCHMARK_OSS_BASE%/}/MME_Benchmark_release_version}}"
MME_HF_ROOT="${MME_HF_ROOT:-${BENCHMARK_DATA_ROOT}/MME_hf}"
MME_HF_OSS_URI="${MME_HF_OSS_URI:-${BENCHMARK_OSS_BASE:+${BENCHMARK_OSS_BASE%/}/MME_hf}}"
MME_HF_DATASET="${MME_HF_DATASET:-lmms-lab/MME}"
MME_HF_SPLIT="${MME_HF_SPLIT:-test}"
MME_MODELSCOPE_ID="${MME_MODELSCOPE_ID:-}"
KEEP_BENCHMARK_DATA="${KEEP_BENCHMARK_DATA:-False}"
MME_CATEGORIES="${MME_CATEGORIES:-existence,count,position,color}"
MME_MAX_SAMPLES="${MME_MAX_SAMPLES:-0}"
MME_PARALLEL_WORKERS="${MME_PARALLEL_WORKERS:-64}"
STUDENT_PX="${STUDENT_PX:-}"
TARGET_PX="${TARGET_PX:-448}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
STAGED_DATASET=0
STAGED_DATASET_ROOT=""
STAGED_CONVERT_DIR=""
MME_SOURCE_KIND=""
CONVERTED_MME_JSON=""

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

stage_from_oss() {
    local oss_uri="$1"
    local local_dir="$2"
    if [[ -z "$oss_uri" ]]; then
        return 1
    fi
    echo "Staging classic MME from OSS: ${oss_uri} -> ${local_dir}"
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
    echo "Staging classic MME from ModelScope dataset: ${dataset_id} -> ${local_dir}"
    mkdir -p "$local_dir"
    modelscope download --dataset "$dataset_id" --local_dir "$local_dir"
}

stage_from_huggingface() {
    local dataset_id="$1"
    local local_dir="$2"
    if [[ -z "$dataset_id" ]]; then
        return 1
    fi
    if ! command -v huggingface-cli >/dev/null 2>&1; then
        echo "huggingface-cli not found; skipping HF download." >&2
        return 1
    fi
    echo "Staging classic MME from HuggingFace dataset: ${dataset_id} -> ${local_dir}"
    mkdir -p "$local_dir"
    huggingface-cli download "$dataset_id" --repo-type dataset --local-dir "$local_dir"
}

has_parquet_data() {
    local local_dir="$1"
    find "$local_dir" -type f -name '*.parquet' 2>/dev/null | grep -q .
}

ensure_mme_data() {
    if [[ -n "${MME_JSON:-}" ]]; then
        if [[ -f "$MME_JSON" ]]; then
            MME_SOURCE_KIND="json"
            return
        fi
        echo "Error: MME_JSON does not exist: $MME_JSON" >&2
        exit 1
    fi

    if has_parquet_data "$MME_HF_ROOT"; then
        MME_SOURCE_KIND="hf_root"
        return
    fi

    local first_category="${MME_CATEGORIES%%,*}"
    if [[ -d "${MME_ROOT}/${first_category}" ]]; then
        MME_SOURCE_KIND="official_root"
        return
    fi
    if stage_from_oss "$MME_HF_OSS_URI" "$MME_HF_ROOT"; then
        STAGED_DATASET=1
        STAGED_DATASET_ROOT="$MME_HF_ROOT"
        MME_SOURCE_KIND="hf_root"
    elif stage_from_huggingface "$MME_HF_DATASET" "$MME_HF_ROOT"; then
        STAGED_DATASET=1
        STAGED_DATASET_ROOT="$MME_HF_ROOT"
        MME_SOURCE_KIND="hf_root"
    elif stage_from_oss "$MME_OSS_URI" "$MME_ROOT"; then
        STAGED_DATASET=1
        STAGED_DATASET_ROOT="$MME_ROOT"
        MME_SOURCE_KIND="official_root"
    elif stage_from_modelscope "$MME_MODELSCOPE_ID" "$MME_ROOT"; then
        STAGED_DATASET=1
        STAGED_DATASET_ROOT="$MME_ROOT"
        MME_SOURCE_KIND="official_root"
    fi

    if [[ "$MME_SOURCE_KIND" == "hf_root" ]]; then
        if ! has_parquet_data "$MME_HF_ROOT"; then
            echo "Error: HF MME parquet data not found at ${MME_HF_ROOT}." >&2
            MME_SOURCE_KIND=""
        fi
    fi
    if [[ "$MME_SOURCE_KIND" == "official_root" ]]; then
        if [[ ! -d "${MME_ROOT}/${first_category}" ]]; then
            echo "Error: classic MME official-root data not found at ${MME_ROOT}." >&2
            MME_SOURCE_KIND=""
        fi
    fi
    if [[ -z "$MME_SOURCE_KIND" ]]; then
        echo "Error: classic MME data not found." >&2
        echo "Set MME_JSON, MME_HF_ROOT, MME_HF_OSS_URI, MME_ROOT, MME_OSS_URI, or MME_MODELSCOPE_ID." >&2
        echo "After manual download, you can upload with:" >&2
        echo "  ossutil cp -r ${MME_HF_ROOT}/ oss://<bucket>/<path>/MME_hf/ -f" >&2
        cleanup_dataset
        exit 1
    fi
}

cleanup_dataset() {
    if [[ -n "$STAGED_CONVERT_DIR" && "$KEEP_BENCHMARK_DATA" != "True" && "$KEEP_BENCHMARK_DATA" != "true" ]]; then
        echo "[cleanup] Removing converted MME data: $STAGED_CONVERT_DIR"
        rm -rf "$STAGED_CONVERT_DIR"
    fi
    if [[ "$STAGED_DATASET" -eq 1 && "$KEEP_BENCHMARK_DATA" != "True" && "$KEEP_BENCHMARK_DATA" != "true" ]]; then
        echo "[cleanup] Removing staged MME data: $STAGED_DATASET_ROOT"
        rm -rf "$STAGED_DATASET_ROOT"
    fi
}

ensure_mme_data
mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$(cd "$RES_OPD_ROOT/.." && pwd):${PYTHONPATH:-}"
export VLLM_DISABLE_PROMETHEUS=1

echo "============================================================"
echo " Classic MME Perception Eval"
echo "============================================================"
echo "Model:      $MODEL_PATH"
echo "Categories: $MME_CATEGORIES"
echo "Student:    mode=$DEGRADATION_MODE px=$STUDENT_PX target=$TARGET_PX ratio=$STUDENT_RATIO"
if [[ -n "${MME_JSON:-}" ]]; then
    echo "MME JSON:   $MME_JSON"
elif [[ "$MME_SOURCE_KIND" == "hf_root" ]]; then
    echo "MME HF:     $MME_HF_ROOT"
    echo "MME HF OSS: ${MME_HF_OSS_URI:-<none>}"
else
    echo "MME root:   $MME_ROOT"
    echo "MME OSS:    ${MME_OSS_URI:-<none>}"
fi
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

source_args=()
if [[ "$MME_SOURCE_KIND" == "json" ]]; then
    source_args+=(--mme-json "$MME_JSON")
elif [[ "$MME_SOURCE_KIND" == "hf_root" ]]; then
    SAFE_EXP_NAME="$(echo "${EXPERIMENT_NAME}_${VERSION_TAG}" | tr -c 'A-Za-z0-9_.-' '_')"
    CONVERT_DIR="${BENCHMARK_DATA_ROOT}/MME_hf_converted/${SAFE_EXP_NAME}"
    STAGED_CONVERT_DIR="$CONVERT_DIR"
    CONVERTED_MME_JSON="${CONVERT_DIR}/mme_perception.json"
    "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/convert_mme_hf_to_json.py" \
        --source "$MME_HF_ROOT" \
        --split "$MME_HF_SPLIT" \
        --categories "$MME_CATEGORIES" \
        --output-json "$CONVERTED_MME_JSON" \
        --image-dir "${CONVERT_DIR}/images"
    source_args+=(--mme-json "$CONVERTED_MME_JSON")
elif [[ "$MME_SOURCE_KIND" == "official_root" ]]; then
    source_args+=(--mme-root "$MME_ROOT" --categories "$MME_CATEGORIES")
else
    echo "Error: unsupported MME source kind: $MME_SOURCE_KIND" >&2
    exit 1
fi

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_mme_classic.py" \
    --api-base "http://localhost:$PORT/v1/" \
    --api-key "${OPENAI_API_KEY:-EMPTY}" \
    --model-name "$MODEL_NAME" \
    "${source_args[@]}" \
    --output-dir "$OUTPUT_DIR" \
    --student-px "$STUDENT_PX" \
    --target-px "$TARGET_PX" \
    --degradation-mode "$DEGRADATION_MODE" \
    --student-ratio "$STUDENT_RATIO" \
    --max-samples "$MME_MAX_SAMPLES" \
    --parallel-workers "$MME_PARALLEL_WORKERS"

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/collect_benchmark_summary.py" \
    --output-dir "$OUTPUT_DIR" || true

cleanup
trap - EXIT

echo "MME perception eval complete: $OUTPUT_DIR"

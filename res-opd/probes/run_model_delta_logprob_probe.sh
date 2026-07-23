#!/usr/bin/env bash
# Sharded P2/P3 model-delta logprob probe.
#
# This compares a trained checkpoint against the base model on base-generated
# captions, while also computing the base-original vs base-lowres RKL gate.
#
# Typical launch:
#   BASE_MODEL_PATH=/home/.../Qwen3VL-8B-Instruct \
#   AFTER_OSS_CHECKPOINT=oss://.../global_step_100 \
#   AFTER_NAME=8b_rkl_tr075_s100 \
#   EVAL_RESULTS=/home/.../base/eval_results.jsonl \
#   AFTER_EVAL_RESULTS=/home/.../after/eval_results.jsonl \
#   GPU_IDS=0,1,2,3,4,5,6,7 \
#   bash res-opd/probes/run_model_delta_logprob_probe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

if [[ -f "${RES_OPD_ROOT}/scripts/path_utils.sh" ]]; then
  # shellcheck source=/dev/null
  source "${RES_OPD_ROOT}/scripts/path_utils.sh"
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if declare -F res_opd_default_python_bin >/dev/null 2>&1; then
    PYTHON_BIN="$(res_opd_default_python_bin)"
  else
    for candidate in \
      /home/zhengyanzhao.zyz/.conda/envs/vision-opd/bin/python3 \
      /home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3
    do
      if [[ -x "$candidate" ]]; then
        PYTHON_BIN="$candidate"
        break
      fi
    done
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
if declare -F res_opd_validate_python_bin >/dev/null 2>&1; then
  res_opd_validate_python_bin "$PYTHON_BIN"
fi

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

truthy() {
  case "${1:-}" in
    True|true|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

infer_gpu_ids() {
  if [[ -n "${GPU_IDS:-}" ]]; then
    echo "$GPU_IDS"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local ids
    ids="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, - || true)"
    if [[ -n "$ids" ]]; then
      echo "$ids"
      return
    fi
  fi
  echo "0"
}

ratio_tag() {
  printf "%s" "$1" | sed 's/\./p/g'
}

has_weights() {
  local path="$1"
  [[ -d "$path" ]] || return 1
  find "$path" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -print -quit | grep -q .
}

download_after_checkpoint_once() {
  [[ -n "${AFTER_OSS_CHECKPOINT:-}" ]] || return 0
  if has_weights "$AFTER_MODEL_PATH"; then
    echo "[OSS] AFTER_MODEL_PATH already has weights, skipping download: $AFTER_MODEL_PATH"
    return 0
  fi
  command -v ossutil >/dev/null 2>&1 || fail "ossutil is required for AFTER_OSS_CHECKPOINT"
  mkdir -p "$AFTER_MODEL_PATH"
  echo "[OSS] Downloading after checkpoint:"
  echo "  from: ${AFTER_OSS_CHECKPOINT%/}/"
  echo "  to:   ${AFTER_MODEL_PATH%/}/"
  ossutil cp -r "${AFTER_OSS_CHECKPOINT%/}/" "${AFTER_MODEL_PATH%/}/" -f
  has_weights "$AFTER_MODEL_PATH" || fail "download finished but no weights found in $AFTER_MODEL_PATH"
}

cleanup_downloaded_after_checkpoint() {
  [[ -n "${AFTER_OSS_CHECKPOINT:-}" ]] || return 0
  truthy "${CLEANUP_AFTER:-False}" || return 0
  [[ -n "${AFTER_MODEL_PATH:-}" ]] || return 0
  case "$AFTER_MODEL_PATH" in
    "${RES_OPD_ROOT}/checkpoints/probe_model_delta/"*)
      echo "[Cleanup] Removing downloaded after checkpoint: $AFTER_MODEL_PATH"
      rm -rf "$AFTER_MODEL_PATH"
      ;;
    *)
      echo "[Cleanup] Refusing to remove non-probe checkpoint path: $AFTER_MODEL_PATH" >&2
      ;;
  esac
}

DEFAULT_MODEL_ROOT=""
DEFAULT_DATA_ROOT=""
if declare -F res_opd_default_model_root >/dev/null 2>&1; then
  DEFAULT_MODEL_ROOT="$(res_opd_default_model_root)"
else
  DEFAULT_MODEL_ROOT="/home/liuyanlin.lyl/notebook/model/qwen"
fi
if declare -F res_opd_default_data_root >/dev/null 2>&1; then
  DEFAULT_DATA_ROOT="$(res_opd_default_data_root)"
else
  DEFAULT_DATA_ROOT="/home/liuyanlin.lyl/notebook/data"
fi

BASE_MODEL_PATH="${BASE_MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Qwen3VL-8B-Instruct}"
BASE_MODEL_NAME="${BASE_MODEL_NAME:-$(basename "$BASE_MODEL_PATH")}"
AFTER_NAME="${AFTER_NAME:-}"
if [[ -z "$AFTER_NAME" && -n "${AFTER_OSS_CHECKPOINT:-}" ]]; then
  AFTER_NAME="$(basename "$(dirname "${AFTER_OSS_CHECKPOINT%/}")")_$(basename "${AFTER_OSS_CHECKPOINT%/}")"
fi
AFTER_NAME="${AFTER_NAME:-$(basename "${AFTER_MODEL_PATH:-after_model}")}"
AFTER_MODEL_PATH="${AFTER_MODEL_PATH:-${RES_OPD_ROOT}/checkpoints/probe_model_delta/${AFTER_NAME}}"

DATASET_NAME="${DATASET_NAME:-train5000_test1000_original_sr1p0}"
STUDENT_RATIO="${STUDENT_RATIO:-1.0}"
TEACHER_RATIO="${TEACHER_RATIO:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-8}"
TOPK="${TOPK:-20}"
TOP_P="${TOP_P:-0.30}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
OVERWRITE="${OVERWRITE:-False}"
KEEP_SHARDS="${KEEP_SHARDS:-False}"
CLEANUP_AFTER="${CLEANUP_AFTER:-False}"
trap cleanup_downloaded_after_checkpoint EXIT

if [[ -z "${EVAL_RESULTS:-}" ]]; then
  candidates=(
    "${RES_OPD_ROOT}/eval_results/instruct/full/${BASE_MODEL_NAME}/${DATASET_NAME}/eval_results.jsonl"
    "${RES_OPD_ROOT}/eval_results/latest/full/${BASE_MODEL_NAME}/${DATASET_NAME}/eval_results.jsonl"
  )
  for candidate in "${candidates[@]}"; do
    if [[ -s "$candidate" ]]; then
      EVAL_RESULTS="$candidate"
      break
    fi
  done
  if [[ -z "${EVAL_RESULTS:-}" ]]; then
    printf "ERROR: EVAL_RESULTS is unset and no default path exists. Tried:\n" >&2
    printf "  - %s\n" "${candidates[@]}" >&2
    exit 1
  fi
fi

TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
IMAGE_ROOT="${IMAGE_ROOT:-${DEFAULT_DATA_ROOT}/COCO/coco2017val/val2017}"
OUTPUT_DIR="${OUTPUT_DIR:-${RES_OPD_ROOT}/probes/results/model_delta_logprob/${BASE_MODEL_NAME}__${AFTER_NAME}__tr$(ratio_tag "$TEACHER_RATIO")}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/shards}"
LOG_DIR="${LOG_DIR:-${RES_OPD_ROOT}/logs/model_delta_logprob/${BASE_MODEL_NAME}__${AFTER_NAME}__tr$(ratio_tag "$TEACHER_RATIO")}"
MERGED_TRACE="${MERGED_TRACE:-${OUTPUT_DIR}/model_delta_trace.jsonl}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_DIR}/model_delta_summary.json}"
SUMMARY_MD="${SUMMARY_MD:-${OUTPUT_DIR}/model_delta_summary.md}"
EXAMPLES_JSONL="${EXAMPLES_JSONL:-${OUTPUT_DIR}/model_delta_examples.jsonl}"

GPU_IDS="$(infer_gpu_ids)"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_LIST[@]}"
[[ "$NUM_GPUS" -gt 0 ]] || fail "No GPUs available in GPU_IDS=${GPU_IDS}"
NUM_SHARDS="${NUM_SHARDS:-$NUM_GPUS}"

[[ -d "$BASE_MODEL_PATH" ]] || fail "BASE_MODEL_PATH does not exist: $BASE_MODEL_PATH"
[[ -s "$EVAL_RESULTS" ]] || fail "EVAL_RESULTS does not exist or is empty: $EVAL_RESULTS"
[[ -s "$TEST_JSON" ]] || fail "TEST_JSON does not exist or is empty: $TEST_JSON"
if [[ -n "${AFTER_EVAL_RESULTS:-}" ]]; then
  [[ -s "$AFTER_EVAL_RESULTS" ]] || fail "AFTER_EVAL_RESULTS does not exist or is empty: $AFTER_EVAL_RESULTS"
fi

download_after_checkpoint_once
[[ -d "$AFTER_MODEL_PATH" ]] || fail "AFTER_MODEL_PATH does not exist: $AFTER_MODEL_PATH"
has_weights "$AFTER_MODEL_PATH" || fail "AFTER_MODEL_PATH has no model weights: $AFTER_MODEL_PATH"

mkdir -p "$OUTPUT_DIR" "$SHARD_DIR" "$LOG_DIR" "${RES_OPD_ROOT}/logs"

echo "============================================================"
echo " Model-Delta Logprob Probe"
echo "============================================================"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "BASE_MODEL_PATH = $BASE_MODEL_PATH"
echo "AFTER_MODEL_PATH = $AFTER_MODEL_PATH"
echo "AFTER_NAME = $AFTER_NAME"
echo "AFTER_OSS_CHECKPOINT = ${AFTER_OSS_CHECKPOINT:-}"
echo "EVAL_RESULTS = $EVAL_RESULTS"
echo "AFTER_EVAL_RESULTS = ${AFTER_EVAL_RESULTS:-}"
echo "TEST_JSON = $TEST_JSON"
echo "IMAGE_ROOT = $IMAGE_ROOT"
echo "STUDENT_RATIO = $STUDENT_RATIO"
echo "TEACHER_RATIO = $TEACHER_RATIO"
echo "GPU_IDS = $GPU_IDS"
echo "NUM_SHARDS = $NUM_SHARDS"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "MAX_MODEL_LEN = $MAX_MODEL_LEN"
echo "BATCH_SIZE = $BATCH_SIZE"
echo "KL_CHUNK_SIZE = $KL_CHUNK_SIZE"
echo "TOPK = $TOPK"
echo "TOP_P = $TOP_P"
echo "OVERWRITE = $OVERWRITE"
echo "KEEP_SHARDS = $KEEP_SHARDS"
echo "CLEANUP_AFTER = $CLEANUP_AFTER"
echo "============================================================"

COMMON_ARGS=(
  --base-model-path "$BASE_MODEL_PATH"
  --after-model-path "$AFTER_MODEL_PATH"
  --after-name "$AFTER_NAME"
  --eval-results "$EVAL_RESULTS"
  --test-json "$TEST_JSON"
  --image-root "$IMAGE_ROOT"
  --student-ratio "$STUDENT_RATIO"
  --teacher-ratio "$TEACHER_RATIO"
  --degradation-mode original
  --max-model-len "$MAX_MODEL_LEN"
  --batch-size "$BATCH_SIZE"
  --kl-chunk-size "$KL_CHUNK_SIZE"
  --topk "$TOPK"
  --top-p "$TOP_P"
  --torch-dtype "$TORCH_DTYPE"
  --max-samples "$MAX_SAMPLES"
  --num-shards "$NUM_SHARDS"
)

if [[ -n "${AFTER_EVAL_RESULTS:-}" ]]; then
  COMMON_ARGS+=(--after-eval-results "$AFTER_EVAL_RESULTS")
fi
if truthy "$OVERWRITE"; then
  COMMON_ARGS+=(--overwrite)
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$((shard % NUM_GPUS))]}"
  shard_jsonl="${SHARD_DIR}/shard_${shard}.jsonl"
  shard_summary_json="${SHARD_DIR}/shard_${shard}_summary.json"
  shard_summary_md="${SHARD_DIR}/shard_${shard}_summary.md"
  shard_examples_jsonl="${SHARD_DIR}/shard_${shard}_examples.jsonl"
  shard_log="${LOG_DIR}/shard_${shard}.log"
  echo "[Shard ${shard}/${NUM_SHARDS}] gpu=${gpu} output=${shard_jsonl}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON_BIN" "${SCRIPT_DIR}/probe_model_delta_logprob.py" \
      "${COMMON_ARGS[@]}" \
      --shard-index "$shard" \
      --output-jsonl "$shard_jsonl" \
      --summary-json "$shard_summary_json" \
      --summary-md "$shard_summary_md" \
      --examples-jsonl "$shard_examples_jsonl" \
      --score-only
  ) > "$shard_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "ERROR: one or more shards failed. Last shard logs:" >&2
  for log in "${LOG_DIR}"/shard_*.log; do
    echo "===== ${log} =====" >&2
    tail -n 80 "$log" >&2 || true
  done
  cleanup_downloaded_after_checkpoint
  exit 1
fi

tmp_merged="${MERGED_TRACE}.tmp"
: > "$tmp_merged"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_jsonl="${SHARD_DIR}/shard_${shard}.jsonl"
  [[ -s "$shard_jsonl" ]] || fail "Shard output is missing or empty: $shard_jsonl"
  cat "$shard_jsonl" >> "$tmp_merged"
done
mv "$tmp_merged" "$MERGED_TRACE"
echo "Merged trace: $MERGED_TRACE"

ANALYZE_ARGS=(
  --base-model-path "$BASE_MODEL_PATH"
  --after-model-path "$AFTER_MODEL_PATH"
  --after-name "$AFTER_NAME"
  --eval-results "$EVAL_RESULTS"
  --test-json "$TEST_JSON"
  --image-root "$IMAGE_ROOT"
  --student-ratio "$STUDENT_RATIO"
  --teacher-ratio "$TEACHER_RATIO"
  --topk "$TOPK"
  --top-p "$TOP_P"
  --output-jsonl "$MERGED_TRACE"
  --summary-json "$SUMMARY_JSON"
  --summary-md "$SUMMARY_MD"
  --examples-jsonl "$EXAMPLES_JSONL"
  --analyze-only
)
if [[ -n "${AFTER_EVAL_RESULTS:-}" ]]; then
  ANALYZE_ARGS+=(--after-eval-results "$AFTER_EVAL_RESULTS")
fi

"$PYTHON_BIN" "${SCRIPT_DIR}/probe_model_delta_logprob.py" "${ANALYZE_ARGS[@]}"

if ! truthy "$KEEP_SHARDS"; then
  rm -f "${SHARD_DIR}"/shard_*.jsonl
fi
cleanup_downloaded_after_checkpoint

echo "Done."
echo "Summary: ${SUMMARY_MD}"
echo "Examples: ${EXAMPLES_JSONL}"

#!/usr/bin/env bash
# Sharded base-model dual-view KL/RKL probe.
#
# This is the clean design-motivation diagnostic for RiskMask:
#   caption source: base model original-image eval_results
#   student dist:   base model + original image
#   teacher dist:   same base model + low-res image
#
# Typical 8B launch:
#   MODEL_PATH=/home/.../Qwen3VL-8B-Instruct \
#   TEACHER_RATIO=0.75 \
#   GPU_IDS=0,1,2,3,4,5,6,7 \
#   bash res-opd/probes/run_base_dual_view_kl_probe.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    /home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 \
    /home/zhengyanzhao.zyz/.conda/envs/vision-opd/bin/python3
  do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

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

MODEL_PATH="${MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-8B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
DATASET_NAME="${DATASET_NAME:-train5000_test1000_original_sr1p0}"
STUDENT_RATIO="${STUDENT_RATIO:-1.0}"
TEACHER_RATIO="${TEACHER_RATIO:-0.75}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BATCH_SIZE="${BATCH_SIZE:-1}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-16}"
TOPK="${TOPK:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TOP_P="${TOP_P:-0.30}"
RANK_SCOPE="${RANK_SCOPE:-global}"
OVERWRITE="${OVERWRITE:-False}"
KEEP_SHARDS="${KEEP_SHARDS:-False}"

GPU_IDS="$(infer_gpu_ids)"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_LIST[@]}"
[[ "$NUM_GPUS" -gt 0 ]] || fail "No GPUs available in GPU_IDS=${GPU_IDS}"
NUM_SHARDS="${NUM_SHARDS:-$NUM_GPUS}"

if [[ -z "${EVAL_RESULTS:-}" ]]; then
  CANDIDATE_EVAL_RESULTS=(
    "${RES_OPD_ROOT}/eval_results/instruct/full/${MODEL_NAME}/${DATASET_NAME}/eval_results.jsonl"
    "${RES_OPD_ROOT}/eval_results/latest/full/${MODEL_NAME}/${DATASET_NAME}/eval_results.jsonl"
  )
  for candidate in "${CANDIDATE_EVAL_RESULTS[@]}"; do
    if [[ -s "$candidate" ]]; then
      EVAL_RESULTS="$candidate"
      break
    fi
  done
  if [[ -z "${EVAL_RESULTS:-}" ]]; then
    printf "ERROR: EVAL_RESULTS was not set and no default path exists. Tried:\n" >&2
    printf "  - %s\n" "${CANDIDATE_EVAL_RESULTS[@]}" >&2
    exit 1
  fi
fi

TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/liuyanlin.lyl/notebook/data/COCO/coco2017val/val2017}"
OUTPUT_DIR="${OUTPUT_DIR:-${RES_OPD_ROOT}/probes/results/base_dual_view_kl_probe/${MODEL_NAME}_sr$(ratio_tag "$STUDENT_RATIO")_tr$(ratio_tag "$TEACHER_RATIO")}"
SHARD_DIR="${SHARD_DIR:-${OUTPUT_DIR}/shards}"
LOG_DIR="${LOG_DIR:-${RES_OPD_ROOT}/logs/base_dual_view_kl_probe/${MODEL_NAME}_sr$(ratio_tag "$STUDENT_RATIO")_tr$(ratio_tag "$TEACHER_RATIO")}"
MERGED_TRACE="${MERGED_TRACE:-${OUTPUT_DIR}/pair_kl_trace.jsonl}"
PAIR_SUMMARY_JSON="${PAIR_SUMMARY_JSON:-${OUTPUT_DIR}/pair_kl_summary.json}"
PAIR_SUMMARY_MD="${PAIR_SUMMARY_MD:-${OUTPUT_DIR}/pair_kl_summary.md}"

[[ -d "$MODEL_PATH" ]] || fail "MODEL_PATH does not exist: $MODEL_PATH"
[[ -s "$EVAL_RESULTS" ]] || fail "EVAL_RESULTS does not exist or is empty: $EVAL_RESULTS"
[[ -s "$TEST_JSON" ]] || fail "TEST_JSON does not exist or is empty: $TEST_JSON"

mkdir -p "$OUTPUT_DIR" "$SHARD_DIR" "$LOG_DIR" "${RES_OPD_ROOT}/logs"

echo "============================================================"
echo " Base Dual-View KL/RKL Probe"
echo "============================================================"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "MODEL_PATH = $MODEL_PATH"
echo "MODEL_NAME = $MODEL_NAME"
echo "EVAL_RESULTS = $EVAL_RESULTS"
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
echo "RANK_SCOPE = $RANK_SCOPE"
echo "OVERWRITE = $OVERWRITE"
echo "KEEP_SHARDS = $KEEP_SHARDS"
echo "============================================================"

COMMON_ARGS=(
  --pair-mode dual_view_same_model
  --model-path "$MODEL_PATH"
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
  --torch-dtype "$TORCH_DTYPE"
  --max-samples "$MAX_SAMPLES"
  --num-shards "$NUM_SHARDS"
)

if truthy "$OVERWRITE"; then
  COMMON_ARGS+=(--overwrite)
fi

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${GPU_LIST[$((shard % NUM_GPUS))]}"
  shard_jsonl="${SHARD_DIR}/shard_${shard}.jsonl"
  shard_summary_json="${SHARD_DIR}/shard_${shard}_summary.json"
  shard_summary_md="${SHARD_DIR}/shard_${shard}_summary.md"
  shard_log="${LOG_DIR}/shard_${shard}.log"
  echo "[Shard ${shard}/${NUM_SHARDS}] gpu=${gpu} output=${shard_jsonl}"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    exec "$PYTHON_BIN" "${SCRIPT_DIR}/probe_same_image_rkl_signal.py" \
      "${COMMON_ARGS[@]}" \
      --shard-index "$shard" \
      --output-jsonl "$shard_jsonl" \
      --summary-json "$shard_summary_json" \
      --summary-md "$shard_summary_md" \
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

"$PYTHON_BIN" "${SCRIPT_DIR}/probe_same_image_rkl_signal.py" \
  --pair-mode dual_view_same_model \
  --model-path "$MODEL_PATH" \
  --eval-results "$EVAL_RESULTS" \
  --test-json "$TEST_JSON" \
  --output-jsonl "$MERGED_TRACE" \
  --summary-json "$PAIR_SUMMARY_JSON" \
  --summary-md "$PAIR_SUMMARY_MD" \
  --analyze-only

TRACE_JSONL="$MERGED_TRACE" \
OUTPUT_DIR="$OUTPUT_DIR" \
TOP_P="$TOP_P" \
RANK_SCOPE="$RANK_SCOPE" \
PYTHON_BIN="$PYTHON_BIN" \
bash "${SCRIPT_DIR}/run_riskmask_selection_enrichment.sh"

if ! truthy "$KEEP_SHARDS"; then
  rm -f "${SHARD_DIR}"/shard_*.jsonl
fi

echo "Done."
echo "Pair summary: ${PAIR_SUMMARY_MD}"
echo "Enrichment summary: ${OUTPUT_DIR}/riskmask_selection_enrichment_summary.md"

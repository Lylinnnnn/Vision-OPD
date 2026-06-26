#!/usr/bin/env bash
# Run targeted POPE over object changes from compare_eval_result_objects.py.
#
# Typical launch for the default base vs tr1.0 RKL comparison:
#
#   tmux new-session -d -s targeted_pope_tr10 \
#     "GPU_IDS=0,1,2,3,4,5,6,7 bash res-opd/probes/run_targeted_pope_from_eval_compare.sh 2>&1 | tee res-opd/logs/targeted_pope_tr10.log"
#
# Resume only tr1.0 RKL and summary after base is complete:
#
#   TARGETED_TASKS=tr10,summary bash res-opd/probes/run_targeted_pope_from_eval_compare.sh
#
# Generic variant launch:
#
#   OTHER_RUN_NAME=tr075_rkl \
#   OTHER_CKPT_EXP=Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-full5k-e1 \
#   TARGETED_TASKS=all bash res-opd/probes/run_targeted_pope_from_eval_compare.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"

# Match the probe/training launchers: prefer conda-bundled cuDNN.
CONDA_CUDNN_LIB="$("$PYTHON_BIN" -c "
import os, sys
try:
    import nvidia.cudnn
    p = getattr(nvidia.cudnn, '__file__', None)
    if p:
        d = os.path.join(os.path.dirname(p), 'lib')
        if os.path.isdir(d):
            print(d); sys.exit(0)
except Exception:
    pass
try:
    import nvidia
    d = os.path.join(nvidia.__path__[0], 'cudnn', 'lib')
    if os.path.isdir(d):
        print(d); sys.exit(0)
except Exception:
    pass
" 2>/dev/null || true)"
if [[ -z "$CONDA_CUDNN_LIB" || ! -d "$CONDA_CUDNN_LIB" ]]; then
  _FALLBACK="$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages/nvidia/cudnn/lib"
  if [[ -d "$_FALLBACK" ]]; then
    CONDA_CUDNN_LIB="$_FALLBACK"
  fi
fi
if [[ -n "$CONDA_CUDNN_LIB" && -d "$CONDA_CUDNN_LIB" ]]; then
  export LD_LIBRARY_PATH="${CONDA_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

truthy() {
  case "${1:-}" in
    True|true|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

task_enabled() {
  local task="$1"
  local requested="${TARGETED_TASKS:-all}"
  if [[ "$requested" == "all" || "$requested" == "ALL" ]]; then
    return 0
  fi
  local -a _targeted_tasks
  IFS=',' read -r -a _targeted_tasks <<< "$requested"
  local item
  for item in "${_targeted_tasks[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ "$item" == "$task" || ( "$task" == "other" && "$item" == "${OTHER_TASK:-other}" ) ]]; then
      return 0
    fi
  done
  return 1
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

checkpoint_has_weights() {
  local path="$1"
  [[ -d "$path" ]] || return 1
  find "$path" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' \) -print -quit | grep -q .
}

download_checkpoint_from_oss() {
  local oss_path="$1"
  local local_path="$2"
  if checkpoint_has_weights "$local_path"; then
    echo "[OSS] Checkpoint already exists at ${local_path}; reusing it."
    return 1
  fi
  if ! command -v ossutil >/dev/null 2>&1; then
    echo "ERROR: ossutil is required to download ${oss_path}" >&2
    exit 1
  fi
  echo "[OSS] Downloading checkpoint: ${oss_path} -> ${local_path}"
  mkdir -p "$local_path"
  ossutil cp -r "${oss_path%/}/" "${local_path%/}/" -f
  if ! checkpoint_has_weights "$local_path"; then
    echo "ERROR: downloaded checkpoint has no weights: ${local_path}" >&2
    exit 1
  fi
  return 0
}

get_oss_name() {
  local ckpt_dir_name="$1"
  local suffix
  suffix="${ckpt_dir_name#Res-OPD-Qwen3VL-2B-Instruct-}"
  if [[ "$suffix" == "$ckpt_dir_name" ]]; then
    suffix="$ckpt_dir_name"
  fi
  local epoch_tag=""
  if [[ "$suffix" =~ ^(.+)-(e[0-9]+)$ ]]; then
    suffix="${BASH_REMATCH[1]}"
    epoch_tag="-${BASH_REMATCH[2]}"
  fi
  echo "ResOPD_${suffix//-/_}${epoch_tag}"
}

BASE_MODEL="${BASE_MODEL:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}"
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
BASE_RESULTS="${BASE_RESULTS:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct/train5000_test1000_original_sr1p0/eval_results.jsonl}"

OTHER_RUN_NAME="${OTHER_RUN_NAME:-${TR10_RUN_NAME:-tr10_rkl}}"
OTHER_TASK="${OTHER_TASK:-${TR10_TASK:-tr10}}"
OTHER_CKPT_EXP="${OTHER_CKPT_EXP:-${TR10_CKPT_EXP:-Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1}}"
OTHER_STEP="${OTHER_STEP:-${TR10_STEP:-global_step_39}}"
OTHER_EVAL_EXP="${OTHER_EVAL_EXP:-${TR10_EVAL_EXP:-${OTHER_CKPT_EXP}_${OTHER_STEP}}}"
OTHER_LOCAL="${OTHER_LOCAL:-${TR10_LOCAL:-${RES_OPD_ROOT}/tmp_checkpoints/${OTHER_CKPT_EXP}/${OTHER_STEP}}}"
OTHER_OSS_NAME="${OTHER_OSS_NAME:-${TR10_OSS_NAME:-$(get_oss_name "$OTHER_CKPT_EXP")}}"
OTHER_OSS="${OTHER_OSS:-${TR10_OSS:-${OSS_BASE%/}/${OTHER_OSS_NAME}/${OTHER_STEP}}}"
OTHER_RESULTS="${OTHER_RESULTS:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/${OTHER_EVAL_EXP}/train5000_test1000_original_sr1p0/eval_results.jsonl}"

COMPARISON_DIR="${COMPARISON_DIR:-${RES_OPD_ROOT}/probes/results/same_image_rkl_signal/eval_compare_base_vs_${OTHER_RUN_NAME}}"
COMPARISON_JSON="${COMPARISON_JSON:-${COMPARISON_DIR}/eval_result_object_comparison.json}"
COMPARISON_MD="${COMPARISON_MD:-${COMPARISON_DIR}/eval_result_object_comparison.md}"
BUILD_COMPARISON="${BUILD_COMPARISON:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-${RES_OPD_ROOT}/probes/results/targeted_pope_base_vs_${OTHER_RUN_NAME}}"
SAMPLES_JSONL="${SAMPLES_JSONL:-${OUTPUT_DIR}/targeted_pope_samples.jsonl}"
TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
TARGETED_BUCKETS="${TARGETED_BUCKETS:-removed_hallucinated,added_hallucinated,removed_correct,added_correct}"
MAX_PER_BUCKET="${MAX_PER_BUCKET:-0}"

GPU_IDS="$(infer_gpu_ids)"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-$GPU_IDS}"
IFS=',' read -r -a VLLM_GPU_LIST <<< "$VLLM_GPU_IDS"
VLLM_NUM_GPUS="${#VLLM_GPU_LIST[@]}"
if [[ "$VLLM_NUM_GPUS" -le 0 ]]; then
  VLLM_NUM_GPUS=1
fi
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-$VLLM_NUM_GPUS}"
BASE_VLLM_PORT="${BASE_VLLM_PORT:-${VLLM_BASE_PORT:-${VLLM_PORT:-8027}}}"
OTHER_VLLM_PORT="${OTHER_VLLM_PORT:-${TR10_VLLM_PORT:-$((BASE_VLLM_PORT + 1))}}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-9728}"
VLLM_PARALLEL_WORKERS="${VLLM_PARALLEL_WORKERS:-64}"
TARGETED_MAX_NEW_TOKENS="${TARGETED_MAX_NEW_TOKENS:-4}"
TARGETED_SAVE_LOGPROBS="${TARGETED_SAVE_LOGPROBS:-True}"
TARGETED_TASKS="${TARGETED_TASKS:-all}"
SKIP_COMPLETED="${SKIP_COMPLETED:-True}"
FORCE_TARGETED="${FORCE_TARGETED:-False}"
ALLOW_PARTIAL_SUMMARY="${ALLOW_PARTIAL_SUMMARY:-False}"
CLEANUP_AFTER="${CLEANUP_AFTER:-False}"

mkdir -p "${RES_OPD_ROOT}/logs" "$OUTPUT_DIR"

VLLM_PID=""
CURRENT_VLLM_PORT=""
wait_port_released() {
  local port="$1"
  local max_wait="${2:-60}"
  local i
  for i in $(seq 1 "$max_wait"); do
    if ! curl -s "http://localhost:${port}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

cleanup_vllm_server() {
  local port="${CURRENT_VLLM_PORT:-}"
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[vLLM] Shutting down server pid=${VLLM_PID}"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
    if [[ -n "$port" ]]; then
      if ! wait_port_released "$port" 60; then
        echo "[vLLM] WARNING: port ${port} still responds after shutdown; continuing because later tasks use their own ports."
      fi
    fi
  fi
  VLLM_PID=""
  CURRENT_VLLM_PORT=""
}
trap cleanup_vllm_server EXIT

start_vllm_server() {
  local model_path="$1"
  local served_model_name="$2"
  local log_path="$3"
  local port="$4"

  cleanup_vllm_server
  if curl -s "http://localhost:${port}/health" >/dev/null 2>&1; then
    echo "ERROR: vLLM port ${port} is already serving /health; set BASE_VLLM_PORT/OTHER_VLLM_PORT to a free port." >&2
    exit 1
  fi

  echo "[vLLM] Starting server model=${model_path} name=${served_model_name}"
  echo "[vLLM] port=${port} gpu_ids=${VLLM_GPU_IDS} tensor_parallel=${VLLM_TENSOR_PARALLEL_SIZE}"
  export VLLM_DISABLE_PROMETHEUS=1
  export VLLM_USE_V1=1
  unset VLLM_ATTENTION_BACKEND
  (
    export CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served_model_name" \
      --trust-remote-code \
      --port "$port" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
      --disable-frontend-multiprocessing
  ) > "$log_path" 2>&1 &
  VLLM_PID=$!
  CURRENT_VLLM_PORT="$port"

  for i in $(seq 1 300); do
    if curl -s "http://localhost:${port}/health" >/dev/null 2>&1; then
      echo "[vLLM] Ready after ${i}s"
      return
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
      echo "ERROR: vLLM exited unexpectedly. Last log lines:" >&2
      tail -n 120 "$log_path" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "ERROR: vLLM failed to become ready. Last log lines:" >&2
  tail -n 120 "$log_path" >&2 || true
  exit 1
}

answers_complete() {
  local samples_path="$1"
  local answers_path="$2"
  local label="$3"
  "$PYTHON_BIN" - "$samples_path" "$answers_path" "$label" <<'PY'
import json
import os
import sys

samples_path, answers_path, label = sys.argv[1:4]

def load_jsonl(path):
    rows = []
    if not path or not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

samples = load_jsonl(samples_path)
answers = load_jsonl(answers_path)
sample_uids = [row.get("sample_uid") for row in samples if row.get("sample_uid")]
answer_by_uid = {}
for row in answers:
    uid = row.get("sample_uid")
    if uid:
        answer_by_uid[uid] = row

missing = [uid for uid in sample_uids if uid not in answer_by_uid]
bad = []
for uid in sample_uids:
    row = answer_by_uid.get(uid)
    if not row:
        continue
    text = str(row.get("model_answer", "")).strip()
    if not text or text.startswith("[ERROR]"):
        bad.append(uid)

complete = bool(sample_uids) and not missing and not bad
print(
    f"[SkipCheck] {label}: samples={len(sample_uids)} "
    f"answers={len(answer_by_uid)} missing={len(missing)} bad={len(bad)} "
    f"complete={complete}"
)
sys.exit(0 if complete else 1)
PY
}

should_skip_answers() {
  local label="$1"
  local answers_path="$2"
  if truthy "$FORCE_TARGETED" || ! truthy "$SKIP_COMPLETED"; then
    return 1
  fi
  if answers_complete "$samples_path" "$answers_path" "$label"; then
    echo "[Skip] ${label} answers are complete; not starting vLLM."
    return 0
  fi
  return 1
}

ensure_complete_for_summary() {
  local label="$1"
  local answers_path="$2"
  if truthy "$ALLOW_PARTIAL_SUMMARY"; then
    return 0
  fi
  if answers_complete "$samples_path" "$answers_path" "$label"; then
    return 0
  fi
  echo "ERROR: ${label} answers are incomplete; run TARGETED_TASKS=${label} first or set ALLOW_PARTIAL_SUMMARY=True." >&2
  return 1
}

maybe_build_comparison() {
  local should_build=False
  if [[ "$BUILD_COMPARISON" == "auto" ]]; then
    if [[ ! -s "$COMPARISON_JSON" ]]; then
      should_build=True
    fi
  elif truthy "$BUILD_COMPARISON"; then
    should_build=True
  fi

  if ! truthy "$should_build"; then
    echo "[compare] Reusing comparison JSON: $COMPARISON_JSON"
    return 0
  fi

  if [[ ! -f "$BASE_RESULTS" ]]; then
    echo "ERROR: base eval_results missing: $BASE_RESULTS" >&2
    exit 1
  fi
  if [[ ! -f "$OTHER_RESULTS" ]]; then
    echo "ERROR: ${OTHER_RUN_NAME} eval_results missing: $OTHER_RESULTS" >&2
    exit 1
  fi

  echo "[compare] Building base vs ${OTHER_RUN_NAME} object comparison"
  "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/compare_eval_result_objects.py" \
    --base-results "$BASE_RESULTS" \
    --other-results "$OTHER_RESULTS" \
    --base-name "base" \
    --other-name "$OTHER_RUN_NAME" \
    --test-json "$TEST_JSON" \
    --output-json "$COMPARISON_JSON" \
    --output-md "$COMPARISON_MD"
}

samples_path="$SAMPLES_JSONL"
base_answers="${OUTPUT_DIR}/base_answers.jsonl"
other_answers="${OUTPUT_DIR}/${OTHER_RUN_NAME}_answers.jsonl"
other_downloaded=False

common_args=(
  --comparison-json "$COMPARISON_JSON"
  --test-json "$TEST_JSON"
  --output-dir "$OUTPUT_DIR"
  --samples-jsonl "$samples_path"
  --buckets "$TARGETED_BUCKETS"
  --max-per-bucket "$MAX_PER_BUCKET"
)

logprob_args=()
if truthy "$TARGETED_SAVE_LOGPROBS"; then
  logprob_args=(--save-logprobs --top-logprobs 20)
fi

echo "=== Targeted POPE config ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "BASE_RESULTS=$BASE_RESULTS"
echo "OTHER_RUN_NAME=$OTHER_RUN_NAME"
echo "OTHER_TASK=$OTHER_TASK"
echo "OTHER_CKPT_EXP=$OTHER_CKPT_EXP"
echo "OTHER_EVAL_EXP=$OTHER_EVAL_EXP"
echo "OTHER_LOCAL=$OTHER_LOCAL"
echo "OTHER_OSS=$OTHER_OSS"
echo "OTHER_RESULTS=$OTHER_RESULTS"
echo "COMPARISON_JSON=$COMPARISON_JSON"
echo "BUILD_COMPARISON=$BUILD_COMPARISON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "SAMPLES_JSONL=$samples_path"
echo "TEST_JSON=$TEST_JSON"
echo "TARGETED_BUCKETS=$TARGETED_BUCKETS MAX_PER_BUCKET=$MAX_PER_BUCKET"
echo "TARGETED_TASKS=$TARGETED_TASKS SKIP_COMPLETED=$SKIP_COMPLETED FORCE_TARGETED=$FORCE_TARGETED"
echo "VLLM_GPU_IDS=$VLLM_GPU_IDS VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE"
echo "BASE_VLLM_PORT=$BASE_VLLM_PORT OTHER_VLLM_PORT=$OTHER_VLLM_PORT"
echo

maybe_build_comparison

if task_enabled build || [[ ! -s "$samples_path" ]]; then
  echo "[0/3] Build targeted POPE samples"
  "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
    "${common_args[@]}" \
    --build-only
else
  echo "[0/3] Reusing targeted POPE samples: $samples_path"
fi

if task_enabled base; then
  echo
  echo "[1/3] Targeted POPE for base"
  if should_skip_answers "base" "$base_answers"; then
    :
  else
    start_vllm_server "$BASE_MODEL" "Qwen3VL-2B-Instruct" "${RES_OPD_ROOT}/logs/vllm_targeted_pope_base.log" "$BASE_VLLM_PORT"
    "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
      "${common_args[@]}" \
      --api-base "http://localhost:${BASE_VLLM_PORT}/v1/" \
      --model-name "Qwen3VL-2B-Instruct" \
      --run-name "base" \
      --answers-jsonl "$base_answers" \
      --parallel-workers "$VLLM_PARALLEL_WORKERS" \
      --max-new-tokens "$TARGETED_MAX_NEW_TOKENS" \
      "${logprob_args[@]}"
    cleanup_vllm_server
  fi
else
  echo
  echo "[1/3] Skipping base because TARGETED_TASKS=$TARGETED_TASKS"
fi

if task_enabled other; then
  echo
  echo "[2/3] Targeted POPE for ${OTHER_RUN_NAME}"
  if should_skip_answers "$OTHER_TASK" "$other_answers"; then
    :
  else
    if download_checkpoint_from_oss "$OTHER_OSS" "$OTHER_LOCAL"; then
      other_downloaded=True
    fi
    start_vllm_server "$OTHER_LOCAL" "$OTHER_EVAL_EXP" "${RES_OPD_ROOT}/logs/vllm_targeted_pope_${OTHER_RUN_NAME}.log" "$OTHER_VLLM_PORT"
    "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
      "${common_args[@]}" \
      --api-base "http://localhost:${OTHER_VLLM_PORT}/v1/" \
      --model-name "$OTHER_EVAL_EXP" \
      --run-name "$OTHER_RUN_NAME" \
      --answers-jsonl "$other_answers" \
      --parallel-workers "$VLLM_PARALLEL_WORKERS" \
      --max-new-tokens "$TARGETED_MAX_NEW_TOKENS" \
      "${logprob_args[@]}"
    cleanup_vllm_server
  fi
else
  echo
  echo "[2/3] Skipping ${OTHER_RUN_NAME} because TARGETED_TASKS=$TARGETED_TASKS"
fi

if task_enabled summary; then
  echo
  echo "[3/3] Paired targeted POPE summary"
  ensure_complete_for_summary "base" "$base_answers"
  ensure_complete_for_summary "$OTHER_TASK" "$other_answers"
  "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
    "${common_args[@]}" \
    --base-answers "$base_answers" \
    --other-answers "$other_answers" \
    --base-name "base" \
    --other-name "$OTHER_RUN_NAME" \
    --summary-json "${OUTPUT_DIR}/targeted_pope_summary.json" \
    --summary-md "${OUTPUT_DIR}/targeted_pope_summary.md"
else
  echo
  echo "[3/3] Skipping summary because TARGETED_TASKS=$TARGETED_TASKS"
fi

if truthy "$CLEANUP_AFTER" && truthy "$other_downloaded"; then
  echo "[Cleanup] Removing checkpoint downloaded by this launcher: $OTHER_LOCAL"
  rm -rf "$OTHER_LOCAL"
fi

echo
echo "Targeted POPE finished:"
echo "  samples: $samples_path"
echo "  base:    $base_answers"
echo "  other:   $other_answers"
echo "  summary: ${OUTPUT_DIR}/targeted_pope_summary.md"

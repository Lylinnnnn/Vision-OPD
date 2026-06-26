#!/usr/bin/env bash
# Run targeted POPE over object changes from compare_eval_result_objects.py.
#
# Typical launch:
#
#   tmux new-session -d -s targeted_pope_tr10 \
#     "GPU_IDS=0,1,2,3,4,5,6,7 bash res-opd/probes/run_targeted_pope_from_eval_compare.sh 2>&1 | tee res-opd/logs/targeted_pope_tr10.log"

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
TR10_CKPT_EXP="${TR10_CKPT_EXP:-Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1}"
TR10_STEP="${TR10_STEP:-global_step_39}"
TR10_EVAL_EXP="${TR10_EVAL_EXP:-${TR10_CKPT_EXP}_${TR10_STEP}}"
TR10_LOCAL="${TR10_LOCAL:-${RES_OPD_ROOT}/tmp_checkpoints/${TR10_CKPT_EXP}/${TR10_STEP}}"
TR10_OSS_NAME="${TR10_OSS_NAME:-$(get_oss_name "$TR10_CKPT_EXP")}"
TR10_OSS="${TR10_OSS:-${OSS_BASE%/}/${TR10_OSS_NAME}/${TR10_STEP}}"

COMPARISON_JSON="${COMPARISON_JSON:-${RES_OPD_ROOT}/probes/results/same_image_kl_probe_results/6_eval_compare_base_vs_tr10_rkl.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RES_OPD_ROOT}/probes/results/targeted_pope_base_vs_tr10_rkl}"
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
VLLM_PORT="${VLLM_PORT:-8027}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-9728}"
VLLM_PARALLEL_WORKERS="${VLLM_PARALLEL_WORKERS:-64}"
TARGETED_MAX_NEW_TOKENS="${TARGETED_MAX_NEW_TOKENS:-4}"
TARGETED_SAVE_LOGPROBS="${TARGETED_SAVE_LOGPROBS:-True}"
CLEANUP_AFTER="${CLEANUP_AFTER:-False}"

mkdir -p "${RES_OPD_ROOT}/logs" "$OUTPUT_DIR"

VLLM_PID=""
cleanup_vllm_server() {
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[vLLM] Shutting down server pid=${VLLM_PID}"
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=""
}
trap cleanup_vllm_server EXIT

start_vllm_server() {
  local model_path="$1"
  local served_model_name="$2"
  local log_path="$3"

  cleanup_vllm_server
  if curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
    echo "ERROR: vLLM port ${VLLM_PORT} is already serving /health; set VLLM_PORT to a free port." >&2
    exit 1
  fi

  echo "[vLLM] Starting server model=${model_path} name=${served_model_name}"
  echo "[vLLM] port=${VLLM_PORT} gpu_ids=${VLLM_GPU_IDS} tensor_parallel=${VLLM_TENSOR_PARALLEL_SIZE}"
  export VLLM_DISABLE_PROMETHEUS=1
  export VLLM_USE_V1=1
  unset VLLM_ATTENTION_BACKEND
  (
    export CUDA_VISIBLE_DEVICES="$VLLM_GPU_IDS"
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
      --model "$model_path" \
      --served-model-name "$served_model_name" \
      --trust-remote-code \
      --port "$VLLM_PORT" \
      --max-model-len "$VLLM_MAX_MODEL_LEN" \
      --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
      --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
      --disable-frontend-multiprocessing
  ) > "$log_path" 2>&1 &
  VLLM_PID=$!

  for i in $(seq 1 300); do
    if curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
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

common_args=(
  --comparison-json "$COMPARISON_JSON"
  --test-json "$TEST_JSON"
  --output-dir "$OUTPUT_DIR"
  --buckets "$TARGETED_BUCKETS"
  --max-per-bucket "$MAX_PER_BUCKET"
)

logprob_args=()
if truthy "$TARGETED_SAVE_LOGPROBS"; then
  logprob_args=(--save-logprobs --top-logprobs 20)
fi

echo "=== Targeted POPE config ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "TR10_LOCAL=$TR10_LOCAL"
echo "TR10_OSS=$TR10_OSS"
echo "COMPARISON_JSON=$COMPARISON_JSON"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TEST_JSON=$TEST_JSON"
echo "TARGETED_BUCKETS=$TARGETED_BUCKETS MAX_PER_BUCKET=$MAX_PER_BUCKET"
echo "VLLM_GPU_IDS=$VLLM_GPU_IDS VLLM_TENSOR_PARALLEL_SIZE=$VLLM_TENSOR_PARALLEL_SIZE VLLM_PORT=$VLLM_PORT"
echo

"$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
  "${common_args[@]}" \
  --build-only

base_answers="${OUTPUT_DIR}/base_answers.jsonl"
tr10_answers="${OUTPUT_DIR}/tr10_rkl_answers.jsonl"

echo
echo "[1/3] Targeted POPE for base"
start_vllm_server "$BASE_MODEL" "Qwen3VL-2B-Instruct" "${RES_OPD_ROOT}/logs/vllm_targeted_pope_base.log"
"$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
  "${common_args[@]}" \
  --api-base "http://localhost:${VLLM_PORT}/v1/" \
  --model-name "Qwen3VL-2B-Instruct" \
  --run-name "base" \
  --answers-jsonl "$base_answers" \
  --parallel-workers "$VLLM_PARALLEL_WORKERS" \
  --max-new-tokens "$TARGETED_MAX_NEW_TOKENS" \
  "${logprob_args[@]}"
cleanup_vllm_server

echo
echo "[2/3] Targeted POPE for tr1.0 RKL"
tr10_downloaded=False
if download_checkpoint_from_oss "$TR10_OSS" "$TR10_LOCAL"; then
  tr10_downloaded=True
fi
start_vllm_server "$TR10_LOCAL" "$TR10_EVAL_EXP" "${RES_OPD_ROOT}/logs/vllm_targeted_pope_tr10.log"
"$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
  "${common_args[@]}" \
  --api-base "http://localhost:${VLLM_PORT}/v1/" \
  --model-name "$TR10_EVAL_EXP" \
  --run-name "tr10_rkl" \
  --answers-jsonl "$tr10_answers" \
  --parallel-workers "$VLLM_PARALLEL_WORKERS" \
  --max-new-tokens "$TARGETED_MAX_NEW_TOKENS" \
  "${logprob_args[@]}"
cleanup_vllm_server

echo
echo "[3/3] Paired targeted POPE summary"
"$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/targeted_pope_from_eval_compare.py" \
  "${common_args[@]}" \
  --base-answers "$base_answers" \
  --other-answers "$tr10_answers" \
  --base-name "base" \
  --other-name "tr10_rkl" \
  --summary-json "${OUTPUT_DIR}/targeted_pope_summary.json" \
  --summary-md "${OUTPUT_DIR}/targeted_pope_summary.md"

if truthy "$CLEANUP_AFTER" && truthy "$tr10_downloaded"; then
  echo "[Cleanup] Removing checkpoint downloaded by this launcher: $TR10_LOCAL"
  rm -rf "$TR10_LOCAL"
fi

echo
echo "Targeted POPE finished:"
echo "  samples: ${OUTPUT_DIR}/targeted_pope_samples.jsonl"
echo "  base:    $base_answers"
echo "  tr10:    $tr10_answers"
echo "  summary: ${OUTPUT_DIR}/targeted_pope_summary.md"

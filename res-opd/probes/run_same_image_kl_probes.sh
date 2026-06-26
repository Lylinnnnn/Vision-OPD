#!/usr/bin/env bash
# Run same-image / low-res KL probes for Res-OPD.
#
# Typical tmux launch from repo root:
#
#   tmux new-session -d -s probe_same_image_kl \
#     "bash res-opd/probes/run_same_image_kl_probes.sh 2>&1 | tee res-opd/logs/probe_same_image_kl.log"
#
# Optional overrides:
#   BASE_MODEL, BASE_SR10, BASE_SR075_DIR
#   OSS_BASE, TR10_CKPT_EXP, TR10_STEP, TR10_EVAL_EXP, TR10_EVAL, TR10_LOCAL
#   PYTHON_BIN, MAX_SAMPLES, KL_CHUNK_SIZE, TOPK, TORCH_DTYPE
#   PARALLEL_PROBES=True|False, GPU_IDS=0,1,2,3,4,5,6,7, NUM_SHARDS=8
#   OVERWRITE=True|False, PROBE_OUTPUT_ROOT
#   RUN_BASE_SR075_EVAL=True|False|auto
#   RUN_DUPLICATE_SR075=True|False
#   RUN_TR10=True|False
#   CLEANUP_AFTER=True|False

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"

# Match the training launcher: prefer conda-bundled cuDNN over system /lib64.
# nvidia.cudnn.__file__ can be None in some installs, so use multiple fallbacks.
CONDA_CUDNN_LIB="$("$PYTHON_BIN" -c "
import os, sys
# Method 1: nvidia.cudnn.__file__
try:
    import nvidia.cudnn
    p = getattr(nvidia.cudnn, '__file__', None)
    if p:
        d = os.path.join(os.path.dirname(p), 'lib')
        if os.path.isdir(d):
            print(d); sys.exit(0)
except Exception:
    pass
# Method 2: derive from nvidia.__path__
try:
    import nvidia
    d = os.path.join(nvidia.__path__[0], 'cudnn', 'lib')
    if os.path.isdir(d):
        print(d); sys.exit(0)
except Exception:
    pass
" 2>/dev/null || true)"
# Method 3: hardcode known conda env path as last resort
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

# Same OSS naming convention as res-opd/scripts/eval_batch_from_oss.sh expects
# and res-opd/scripts/ckpt_upload_watcher.sh writes:
#   Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1
#   -> ResOPD_orig_sr1.0_tr1.0_a1.0_frozen_rkl_full5k-e1
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
BASE_DIR="${BASE_DIR:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct}"
BASE_SR10="${BASE_SR10:-${BASE_DIR}/train5000_test1000_original_sr1p0/eval_results.jsonl}"
BASE_SR075_DIR="${BASE_SR075_DIR:-${BASE_DIR}/train5000_test1000_original_sr0p75}"
BASE_SR075="${BASE_SR075:-${BASE_SR075_DIR}/eval_results.jsonl}"

OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
TR10_CKPT_EXP="${TR10_CKPT_EXP:-Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1}"
TR10_STEP="${TR10_STEP:-global_step_39}"
TR10_EVAL_EXP="${TR10_EVAL_EXP:-${TR10_CKPT_EXP}_${TR10_STEP}}"
TR10_EVAL="${TR10_EVAL:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/${TR10_EVAL_EXP}/train5000_test1000_original_sr1p0/eval_results.jsonl}"
TR10_LOCAL="${TR10_LOCAL:-${RES_OPD_ROOT}/tmp_checkpoints/${TR10_CKPT_EXP}/${TR10_STEP}}"
TR10_OSS_NAME="${TR10_OSS_NAME:-$(get_oss_name "$TR10_CKPT_EXP")}"
TR10_OSS="${TR10_OSS:-${OSS_BASE%/}/${TR10_OSS_NAME}/${TR10_STEP}}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
BASE_EVAL_MAX_SAMPLES="${BASE_EVAL_MAX_SAMPLES:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-16}"
TOPK="${TOPK:-20}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
CLEANUP_AFTER="${CLEANUP_AFTER:-True}"
OVERWRITE="${OVERWRITE:-False}"
PARALLEL_PROBES="${PARALLEL_PROBES:-True}"
GPU_IDS="$(infer_gpu_ids)"
IFS=',' read -r -a GPU_LIST <<< "$GPU_IDS"
NUM_GPUS="${#GPU_LIST[@]}"
if [[ "$NUM_GPUS" -le 0 ]]; then
  echo "ERROR: no GPUs available in GPU_IDS=${GPU_IDS}" >&2
  exit 1
fi
NUM_SHARDS="${NUM_SHARDS:-$NUM_GPUS}"
PROBE_OUTPUT_ROOT="${PROBE_OUTPUT_ROOT:-${RES_OPD_ROOT}/probes/results/same_image_rkl_signal}"
RUN_BASE_SR075_EVAL="${RUN_BASE_SR075_EVAL:-auto}"
RUN_DUPLICATE_SR075="${RUN_DUPLICATE_SR075:-True}"
RUN_TR10="${RUN_TR10:-True}"

mkdir -p "${RES_OPD_ROOT}/logs" "${RES_OPD_ROOT}/tmp_checkpoints"

overwrite_arg=()
if truthy "$OVERWRITE"; then
  overwrite_arg=(--overwrite)
fi

common_probe_args=(
  --test-json "${RES_OPD_ROOT}/data/test_1000.json"
  --degradation-mode original
  --max-samples "$MAX_SAMPLES"
  --kl-chunk-size "$KL_CHUNK_SIZE"
  --topk "$TOPK"
  --torch-dtype "$TORCH_DTYPE"
  "${overwrite_arg[@]}"
)

run_probe() {
  local label="$1"
  shift
  local output_dir="${PROBE_OUTPUT_ROOT}/${label}"
  local shard_dir="${output_dir}/shards"
  local output_jsonl="${output_dir}/pair_kl_trace.jsonl"
  local summary_json="${output_dir}/pair_kl_object_summary.json"
  local summary_md="${output_dir}/pair_kl_object_summary.md"
  local complete_marker="${output_dir}/.complete"
  local probe_args=("$@")

  mkdir -p "$output_dir" "$shard_dir"

  if ! truthy "$OVERWRITE" \
    && [[ -f "$complete_marker" ]] \
    && [[ -s "$output_jsonl" ]] \
    && [[ -f "$summary_json" ]] \
    && [[ -f "$summary_md" ]]; then
    echo "Reusing complete probe output: ${output_dir}"
    return
  fi

  if truthy "$OVERWRITE"; then
    rm -f "$output_jsonl" "$summary_json" "$summary_md" "$complete_marker"
    rm -f "${shard_dir}"/pair_kl_trace.shard_*.jsonl "${shard_dir}"/shard_*.log 2>/dev/null || true
  fi

  echo "Probe output: ${output_dir}"
  if truthy "$PARALLEL_PROBES"; then
    echo "Running ${label} with ${NUM_SHARDS} shards on GPUs: ${GPU_IDS}"
    local pids=()
    local shard_logs=()
    local shard
    for ((shard = 0; shard < NUM_SHARDS; shard++)); do
      local gpu="${GPU_LIST[$((shard % NUM_GPUS))]}"
      local shard_jsonl="${shard_dir}/pair_kl_trace.shard_${shard}_of_${NUM_SHARDS}.jsonl"
      local shard_log="${shard_dir}/shard_${shard}_gpu_${gpu}.log"
      shard_logs+=("$shard_log")
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
          "${probe_args[@]}" \
          "${common_probe_args[@]}" \
          --num-shards "$NUM_SHARDS" \
          --shard-index "$shard" \
          --output-jsonl "$shard_jsonl" \
          --summary-json "${shard_dir}/summary.shard_${shard}.json" \
          --summary-md "${shard_dir}/summary.shard_${shard}.md" \
          --score-only
      ) > "$shard_log" 2>&1 &
      pids+=("$!")
      echo "  shard ${shard}/${NUM_SHARDS} -> GPU ${gpu}, log=${shard_log}"
    done

    local failed=0
    local idx
    for idx in "${!pids[@]}"; do
      if ! wait "${pids[$idx]}"; then
        echo "ERROR: shard ${idx} failed. Last log lines:" >&2
        tail -n 80 "${shard_logs[$idx]}" >&2 || true
        failed=1
      fi
    done
    if [[ "$failed" -ne 0 ]]; then
      exit 1
    fi

    : > "$output_jsonl"
    for ((shard = 0; shard < NUM_SHARDS; shard++)); do
      local shard_jsonl="${shard_dir}/pair_kl_trace.shard_${shard}_of_${NUM_SHARDS}.jsonl"
      if [[ -f "$shard_jsonl" ]]; then
        cat "$shard_jsonl" >> "$output_jsonl"
      fi
    done

    "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
      "${probe_args[@]}" \
      "${common_probe_args[@]}" \
      --output-jsonl "$output_jsonl" \
      --summary-json "$summary_json" \
      --summary-md "$summary_md" \
      --analyze-only
  else
    "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
      "${probe_args[@]}" \
      "${common_probe_args[@]}" \
      --output-jsonl "$output_jsonl" \
      --summary-json "$summary_json" \
      --summary-md "$summary_md"
  fi

  date > "$complete_marker"
}

echo "=== Same-image KL probe config ==="
echo "VISION_OPD_ROOT=$VISION_OPD_ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "PYTHONPATH=$PYTHONPATH"
echo "CONDA_CUDNN_LIB=${CONDA_CUDNN_LIB:-<unset>}"
echo "BASE_MODEL=$BASE_MODEL"
echo "BASE_SR10=$BASE_SR10"
echo "BASE_SR075=$BASE_SR075"
echo "OSS_BASE=$OSS_BASE"
echo "TR10_CKPT_EXP=$TR10_CKPT_EXP"
echo "TR10_STEP=$TR10_STEP"
echo "TR10_EVAL_EXP=$TR10_EVAL_EXP"
echo "TR10_OSS_NAME=$TR10_OSS_NAME"
echo "TR10_EVAL=$TR10_EVAL"
echo "TR10_LOCAL=$TR10_LOCAL"
echo "TR10_OSS=$TR10_OSS"
echo "MAX_SAMPLES=$MAX_SAMPLES KL_CHUNK_SIZE=$KL_CHUNK_SIZE TOPK=$TOPK"
echo "OVERWRITE=$OVERWRITE PARALLEL_PROBES=$PARALLEL_PROBES GPU_IDS=$GPU_IDS NUM_SHARDS=$NUM_SHARDS"
echo "PROBE_OUTPUT_ROOT=$PROBE_OUTPUT_ROOT"
echo

echo "[0/4] Ensure base sr0.75 eval_results exists"
need_sr075_eval=False
if [[ ! -f "$BASE_SR075" ]]; then
  need_sr075_eval=True
fi
if [[ "$RUN_BASE_SR075_EVAL" == "auto" && "$need_sr075_eval" == "True" ]] || truthy "$RUN_BASE_SR075_EVAL"; then
  mkdir -p "$BASE_SR075_DIR"
  "$PYTHON_BIN" -u "${RES_OPD_ROOT}/eval/eval_chair.py" \
    --model-path "$BASE_MODEL" \
    --test-json "${RES_OPD_ROOT}/data/test_1000.json" \
    --output-dir "$BASE_SR075_DIR" \
    --degradation-mode original \
    --student-ratio 0.75 \
    --max-samples "$BASE_EVAL_MAX_SAMPLES"
else
  echo "Found or skipped sr0.75 eval: $BASE_SR075"
fi
if [[ ! -f "$BASE_SR075" && "$(printf '%s' "$RUN_DUPLICATE_SR075" | tr '[:upper:]' '[:lower:]')" != "false" ]]; then
  echo "ERROR: sr0.75 eval_results missing: $BASE_SR075" >&2
  exit 1
fi

echo
echo "[1/4] duplicate_base on original captions, original image twice"
run_probe "duplicate_base_base_sr10_original_twice" \
  --pair-mode duplicate_base \
  --model-path "$BASE_MODEL" \
  --eval-results "$BASE_SR10" \
  --student-ratio 1.0 \
  --teacher-ratio 1.0 \
  --caption-source-label base_sr10_caption

if truthy "$RUN_DUPLICATE_SR075"; then
  echo
  echo "[2/4] duplicate_base on sr0.75 captions, degraded image twice"
  run_probe "duplicate_base_base_sr075_lowres_twice" \
    --pair-mode duplicate_base \
    --model-path "$BASE_MODEL" \
    --eval-results "$BASE_SR075" \
    --student-ratio 0.75 \
    --teacher-ratio 0.75 \
    --caption-source-label base_sr075_caption
else
  echo
  echo "[2/4] Skipping duplicate_base sr0.75 captions"
fi

echo
echo "[3/4] dual_view_same_model, base original captions, full vs lowres 0.75"
run_probe "dual_view_base_sr10_full_vs_lowres075" \
  --pair-mode dual_view_same_model \
  --model-path "$BASE_MODEL" \
  --eval-results "$BASE_SR10" \
  --student-ratio 1.0 \
  --teacher-ratio 0.75 \
  --caption-source-label base_sr10_caption

if truthy "$RUN_TR10"; then
  if [[ ! -f "$TR10_EVAL" ]]; then
    echo "ERROR: tr1.0 eval_results missing: $TR10_EVAL" >&2
    exit 1
  fi

  tr10_downloaded=False
  if download_checkpoint_from_oss "$TR10_OSS" "$TR10_LOCAL"; then
    tr10_downloaded=True
  fi

  echo
  echo "[4/4] dual_model_same_image, tr1.0 RKL student vs frozen base teacher"
  run_probe "dual_model_tr10_rkl_vs_base_original" \
    --pair-mode dual_model_same_image \
    --student-model-path "$TR10_LOCAL" \
    --teacher-model-path "$BASE_MODEL" \
    --eval-results "$TR10_EVAL" \
    --student-ratio 1.0 \
    --teacher-ratio 1.0 \
    --caption-source-label tr10_rkl_caption

  if truthy "$CLEANUP_AFTER" && truthy "$tr10_downloaded"; then
    echo "[Cleanup] Removing checkpoint downloaded by this launcher: $TR10_LOCAL"
    rm -rf "$TR10_LOCAL"
    tr10_parent="$(dirname "$TR10_LOCAL")"
    if [[ -d "$tr10_parent" ]] && [[ -z "$(ls -A "$tr10_parent" 2>/dev/null)" ]]; then
      rmdir "$tr10_parent" 2>/dev/null || true
    fi
  fi
else
  echo
  echo "[4/4] Skipping tr1.0 dual-model probe"
fi

echo
echo "All same-image KL probes finished."

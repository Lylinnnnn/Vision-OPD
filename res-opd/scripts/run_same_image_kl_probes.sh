#!/usr/bin/env bash
# Run same-image / low-res KL probes for Res-OPD.
#
# Typical tmux launch:
#
#   TR10_OSS=oss://bucket/path/to/global_step_39 \
#   tmux new-session -d -s probe_same_image_kl \
#     "bash res-opd/scripts/run_same_image_kl_probes.sh 2>&1 | tee res-opd/logs/probe_same_image_kl.log"
#
# Required for the tr1.0 dual-model probe:
#   TR10_OSS: OSS path to the merged sr1.0+tr1.0 RKL checkpoint.
#
# Optional overrides:
#   BASE_MODEL, BASE_SR10, BASE_SR075_DIR, TR10_EXP, TR10_EVAL, TR10_LOCAL
#   PYTHON_BIN, MAX_SAMPLES, KL_CHUNK_SIZE, TOPK, TORCH_DTYPE
#   RUN_BASE_SR075_EVAL=True|False|auto
#   RUN_DUPLICATE_SR075=True|False
#   RUN_TR10=True|False
#   CLEANUP_AFTER=True|False

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"
cd "$REPO_ROOT"

truthy() {
  case "${1:-}" in
    True|true|TRUE|1|yes|YES|y|Y) return 0 ;;
    *) return 1 ;;
  esac
}

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL="${BASE_MODEL:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}"
BASE_DIR="${BASE_DIR:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/Qwen3VL-2B-Instruct}"
BASE_SR10="${BASE_SR10:-${BASE_DIR}/train5000_test1000_original_sr1p0/eval_results.jsonl}"
BASE_SR075_DIR="${BASE_SR075_DIR:-${BASE_DIR}/train5000_test1000_original_sr0p75}"
BASE_SR075="${BASE_SR075:-${BASE_SR075_DIR}/eval_results.jsonl}"

TR10_EXP="${TR10_EXP:-Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1_global_step_39}"
TR10_EVAL="${TR10_EVAL:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full/${TR10_EXP}/train5000_test1000_original_sr1p0/eval_results.jsonl}"
TR10_LOCAL="${TR10_LOCAL:-${RES_OPD_ROOT}/tmp_checkpoints/${TR10_EXP}}"
TR10_OSS="${TR10_OSS:-}"

MAX_SAMPLES="${MAX_SAMPLES:-0}"
BASE_EVAL_MAX_SAMPLES="${BASE_EVAL_MAX_SAMPLES:-0}"
KL_CHUNK_SIZE="${KL_CHUNK_SIZE:-16}"
TOPK="${TOPK:-20}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
CLEANUP_AFTER="${CLEANUP_AFTER:-True}"
RUN_BASE_SR075_EVAL="${RUN_BASE_SR075_EVAL:-auto}"
RUN_DUPLICATE_SR075="${RUN_DUPLICATE_SR075:-True}"
RUN_TR10="${RUN_TR10:-True}"

mkdir -p "${RES_OPD_ROOT}/logs" "${RES_OPD_ROOT}/tmp_checkpoints"

cleanup_arg=()
if truthy "$CLEANUP_AFTER"; then
  cleanup_arg=(--cleanup-after)
fi

common_probe_args=(
  --test-json "${RES_OPD_ROOT}/data/test_1000.json"
  --degradation-mode original
  --max-samples "$MAX_SAMPLES"
  --kl-chunk-size "$KL_CHUNK_SIZE"
  --topk "$TOPK"
  --torch-dtype "$TORCH_DTYPE"
  --overwrite
)

echo "=== Same-image KL probe config ==="
echo "BASE_MODEL=$BASE_MODEL"
echo "BASE_SR10=$BASE_SR10"
echo "BASE_SR075=$BASE_SR075"
echo "TR10_EXP=$TR10_EXP"
echo "TR10_EVAL=$TR10_EVAL"
echo "TR10_LOCAL=$TR10_LOCAL"
echo "TR10_OSS=${TR10_OSS:-<unset>}"
echo "MAX_SAMPLES=$MAX_SAMPLES KL_CHUNK_SIZE=$KL_CHUNK_SIZE TOPK=$TOPK"
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
"$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
  --pair-mode duplicate_base \
  --model-path "$BASE_MODEL" \
  --eval-results "$BASE_SR10" \
  --student-ratio 1.0 \
  --teacher-ratio 1.0 \
  --caption-source-label base_sr10_caption \
  "${common_probe_args[@]}"

if truthy "$RUN_DUPLICATE_SR075"; then
  echo
  echo "[2/4] duplicate_base on sr0.75 captions, degraded image twice"
  "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
    --pair-mode duplicate_base \
    --model-path "$BASE_MODEL" \
    --eval-results "$BASE_SR075" \
    --student-ratio 0.75 \
    --teacher-ratio 0.75 \
    --caption-source-label base_sr075_caption \
    "${common_probe_args[@]}"
else
  echo
  echo "[2/4] Skipping duplicate_base sr0.75 captions"
fi

echo
echo "[3/4] dual_view_same_model, base original captions, full vs lowres 0.75"
"$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
  --pair-mode dual_view_same_model \
  --model-path "$BASE_MODEL" \
  --eval-results "$BASE_SR10" \
  --student-ratio 1.0 \
  --teacher-ratio 0.75 \
  --caption-source-label base_sr10_caption \
  "${common_probe_args[@]}"

if truthy "$RUN_TR10"; then
  if [[ -z "$TR10_OSS" ]]; then
    echo "ERROR: RUN_TR10=True requires TR10_OSS=oss://... for the merged tr1.0 RKL checkpoint." >&2
    exit 1
  fi
  if [[ ! -f "$TR10_EVAL" ]]; then
    echo "ERROR: tr1.0 eval_results missing: $TR10_EVAL" >&2
    exit 1
  fi

  echo
  echo "[4/4] dual_model_same_image, tr1.0 RKL student vs frozen base teacher"
  "$PYTHON_BIN" -u "${RES_OPD_ROOT}/probes/probe_same_image_rkl_signal.py" \
    --pair-mode dual_model_same_image \
    --student-model-path "$TR10_LOCAL" \
    --student-oss-checkpoint "$TR10_OSS" \
    --teacher-model-path "$BASE_MODEL" \
    --eval-results "$TR10_EVAL" \
    --student-ratio 1.0 \
    --teacher-ratio 1.0 \
    --caption-source-label tr10_rkl_caption \
    "${cleanup_arg[@]}" \
    "${common_probe_args[@]}"
else
  echo
  echo "[4/4] Skipping tr1.0 dual-model probe"
fi

echo
echo "All same-image KL probes finished."

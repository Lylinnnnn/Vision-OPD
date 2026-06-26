#!/usr/bin/env bash
# Run targeted POPE for multiple base-vs-variant comparisons.
#
# Default variants focus on the current low-res teacher mainline:
#   - tr075_rkl
#   - tr075_rkl_sw
#   - tr075_rkl_sw075
#
# Example:
#
#   tmux new-session -d -s targeted_pope_tr075 \
#     "GPU_IDS=0,1,2,3,4,5,6,7 bash res-opd/probes/run_targeted_pope_variants.sh 2>&1 | tee res-opd/logs/targeted_pope_tr075.log"
#
# To override the variant list:
#
#   TARGETED_VARIANTS=tr075_rkl,tr075_rkl_sw,tr075_rkl_sw075 bash res-opd/probes/run_targeted_pope_variants.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

TARGETED_VARIANTS="${TARGETED_VARIANTS:-tr075_rkl,tr075_rkl_sw,tr075_rkl_sw075}"
TARGETED_STEP="${TARGETED_STEP:-global_step_39}"
TARGETED_VARIANT_BASE_PORT="${TARGETED_VARIANT_BASE_PORT:-8027}"
TARGETED_PORT_STRIDE="${TARGETED_PORT_STRIDE:-2}"
TARGETED_TASKS="${TARGETED_TASKS:-all}"
CLEANUP_AFTER="${CLEANUP_AFTER:-True}"
SEED_BASE_ANSWERS="${SEED_BASE_ANSWERS:-True}"
BASE_ANSWER_CACHE_GLOB="${BASE_ANSWER_CACHE_GLOB:-${RES_OPD_ROOT}/probes/results/targeted_pope_base_vs_*/base_answers.jsonl}"

declare -A EXP_BY_VARIANT=(
  ["tr10_rkl"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1"
  ["tr075_rkl"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-full5k-e1"
  ["tr075_rkl_sw"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw-full5k-e1"
  ["tr075_rkl_sw075"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw075-full5k-e1"
)

declare -A TASK_BY_VARIANT=(
  ["tr10_rkl"]="tr10"
  ["tr075_rkl"]="tr075"
  ["tr075_rkl_sw"]="tr075_sw"
  ["tr075_rkl_sw075"]="tr075_sw075"
)

IFS=',' read -r -a variants <<< "$TARGETED_VARIANTS"

echo "=== Targeted POPE variant batch ==="
echo "TARGETED_VARIANTS=$TARGETED_VARIANTS"
echo "TARGETED_TASKS=$TARGETED_TASKS"
echo "TARGETED_STEP=$TARGETED_STEP"
echo "TARGETED_VARIANT_BASE_PORT=$TARGETED_VARIANT_BASE_PORT TARGETED_PORT_STRIDE=$TARGETED_PORT_STRIDE"
echo "CLEANUP_AFTER=$CLEANUP_AFTER SEED_BASE_ANSWERS=$SEED_BASE_ANSWERS"
echo "BASE_ANSWER_CACHE_GLOB=$BASE_ANSWER_CACHE_GLOB"
echo

idx=0
for raw_variant in "${variants[@]}"; do
  variant="${raw_variant//[[:space:]]/}"
  if [[ -z "$variant" ]]; then
    continue
  fi

  exp="${EXP_BY_VARIANT[$variant]:-}"
  if [[ -z "$exp" ]]; then
    echo "ERROR: unknown TARGETED_VARIANTS item: $variant" >&2
    echo "Supported variants: ${!EXP_BY_VARIANT[*]}" >&2
    exit 1
  fi

  task="${TASK_BY_VARIANT[$variant]:-other}"
  base_port=$((TARGETED_VARIANT_BASE_PORT + idx * TARGETED_PORT_STRIDE))
  other_port=$((base_port + 1))

  echo
  echo "=== [$((idx + 1))/${#variants[@]}] ${variant} ==="
  echo "OTHER_CKPT_EXP=$exp"
  echo "BASE_VLLM_PORT=$base_port OTHER_VLLM_PORT=$other_port"

  OTHER_RUN_NAME="$variant" \
  OTHER_TASK="$task" \
  OTHER_CKPT_EXP="$exp" \
  OTHER_STEP="$TARGETED_STEP" \
  BASE_VLLM_PORT="$base_port" \
  OTHER_VLLM_PORT="$other_port" \
  TARGETED_TASKS="$TARGETED_TASKS" \
  CLEANUP_AFTER="$CLEANUP_AFTER" \
  SEED_BASE_ANSWERS="$SEED_BASE_ANSWERS" \
  BASE_ANSWER_CACHE_GLOB="$BASE_ANSWER_CACHE_GLOB" \
    bash "${RES_OPD_ROOT}/probes/run_targeted_pope_from_eval_compare.sh"

  idx=$((idx + 1))
done

echo
echo "Targeted POPE variant batch finished."

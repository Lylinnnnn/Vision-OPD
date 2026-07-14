#!/bin/bash
# Re-run AMBER only for all Thinking 2B checkpoints with max_new_tokens_generative=4096
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${SCRIPT_DIR}/path_utils.sh"
cd "$RES_OPD_ROOT"

EVAL_BATCH_SCRIPT="${RES_OPD_ROOT}/scripts/eval_batch_from_oss.sh"
SHARDED_EVAL_SCRIPT="${RES_OPD_ROOT}/scripts/eval_after_merge_sharded_8gpu.sh"
LOG_FILE="${RES_OPD_ROOT}/logs/rerun_amber_thinking_2b.log"

export AMBER_MAX_NEW_TOKENS_GENERATIVE=4096

echo "==========================================" | tee "$LOG_FILE"
echo " Re-run AMBER (max_tokens=4096) for Thinking 2B" | tee -a "$LOG_FILE"
echo " Started at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Common args for OSS-based runs
COMMON_ARGS=(
    --eval-mode amber
    --version-tag thinking
    --dataset-version full
    --student-ratio 1.0
    --target-px 448
    --degradation-mode original
)

# --- 1. Baseline: Qwen3-VL-2B-Thinking (local model, use sharded eval directly) ---
echo "" | tee -a "$LOG_FILE"
echo "[1/5] Baseline: Qwen3-VL-2B-Thinking" | tee -a "$LOG_FILE"
BASELINE_MODEL="${BASELINE_MODEL:-$(res_opd_default_model_root)/Qwen3-VL-2B-Thinking}"
if [[ -d "$BASELINE_MODEL" ]]; then
    bash "$SHARDED_EVAL_SCRIPT" "$BASELINE_MODEL" 0 "" amber \
        2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️ Baseline failed" | tee -a "$LOG_FILE"
else
    echo "  ⚠️ Baseline model not found at $BASELINE_MODEL, skipping" | tee -a "$LOG_FILE"
fi

# --- 2. RKL b32 step100 ---
echo "" | tee -a "$LOG_FILE"
echo "[2/5] RKL b32 step100" | tee -a "$LOG_FILE"
bash "$EVAL_BATCH_SCRIPT" \
    --experiment-names "Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-b32-rn4-full5k-e1" \
    --step global_step_100 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️ RKL b32 step100 failed" | tee -a "$LOG_FILE"

# --- 3. RKL b32 step150 ---
echo "" | tee -a "$LOG_FILE"
echo "[3/5] RKL b32 step150" | tee -a "$LOG_FILE"
bash "$EVAL_BATCH_SCRIPT" \
    --experiment-names "Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-b32-rn4-full5k-e1" \
    --step global_step_150 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️ RKL b32 step150 failed" | tee -a "$LOG_FILE"

# --- 4. riskmask-nll-p30 step100 ---
echo "" | tee -a "$LOG_FILE"
echo "[4/5] riskmask-nll-p30 step100" | tee -a "$LOG_FILE"
bash "$EVAL_BATCH_SCRIPT" \
    --experiment-names "Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-riskmask-nll-p30-b32-rn4-full5k-e1" \
    --step global_step_100 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️ riskmask step100 failed" | tee -a "$LOG_FILE"

# --- 5. riskmask-nll-p30 step150 ---
echo "" | tee -a "$LOG_FILE"
echo "[5/5] riskmask-nll-p30 step150" | tee -a "$LOG_FILE"
bash "$EVAL_BATCH_SCRIPT" \
    --experiment-names "Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-riskmask-nll-p30-b32-rn4-full5k-e1" \
    --step global_step_150 \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee -a "$LOG_FILE" || echo "  ⚠️ riskmask step150 failed" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo " All AMBER re-runs completed!" | tee -a "$LOG_FILE"
echo " Finished at: $(date)" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

#!/bin/bash
# =============================================================================
# EMA Baseline Batch - Machine B (3 experiments)
# Student=448, Teacher PX: 300, 112, 0
# All use EMA baseline config with rollout.n=4
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

LOG_DIR="${RES_OPD_ROOT}/logs/ema_baseline_batch"
mkdir -p "$LOG_DIR"

run_experiment() {
    local teacher_px="$1"
    local log_file="${LOG_DIR}/s448_t${teacher_px}_ema.log"

    echo "=========================================="
    echo "Starting: STUDENT_PX=448 TEACHER_PX=${teacher_px} EMA"
    echo "Log: ${log_file}"
    echo "=========================================="

    STUDENT_PX=448 \
    TARGET_PX=448 \
    TEACHER_PX=${teacher_px} \
    TEACHER_MODE=ema \
    ALPHA=0.5 \
    ROLLOUT_N=4 \
    TOTAL_EPOCHS=2 \
    SAVE_FREQ=20 \
    TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$log_file"

    echo "✅ Finished: s448_t${teacher_px}_ema"
    echo ""
}

echo "EMA Baseline Batch - Machine B"
echo "Experiments: t300, t112, t0"
echo "Started at: $(date)"
echo ""

run_experiment 300
run_experiment 112
run_experiment 0

echo "=========================================="
echo "All experiments on Machine B completed!"
echo "Finished at: $(date)"
echo "=========================================="

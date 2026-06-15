#!/bin/bash
# =============================================================================
# Frozen Teacher JSD experiments (v5): s448-t224 and s448-t336
# Purpose: compare JSD vs RKL for the same student-teacher configs
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

LOG_DIR="res-opd/logs/frozen_batch_v5"
mkdir -p "$LOG_DIR"

EXPERIMENT="$1"

if [[ "$EXPERIMENT" == "s448t224_jsd" ]]; then
    echo "[s448-t224 frozen+JSD] Starting at $(date)"
    STUDENT_PX=448 TARGET_PX=448 TEACHER_PX=224 \
        ALPHA=0.5 TEACHER_MODE=frozen \
        SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
        bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s448_t224_frozen_jsd.log"
    echo "[s448-t224 frozen+JSD] DONE at $(date)"

elif [[ "$EXPERIMENT" == "s448t336_jsd" ]]; then
    echo "[s448-t336 frozen+JSD] Starting at $(date)"
    STUDENT_PX=448 TARGET_PX=448 TEACHER_PX=336 \
        ALPHA=0.5 TEACHER_MODE=frozen \
        SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
        bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s448_t336_frozen_jsd.log"
    echo "[s448-t336 frozen+JSD] DONE at $(date)"

else
    echo "Usage: $0 {s448t224_jsd|s448t336_jsd}"
    exit 1
fi

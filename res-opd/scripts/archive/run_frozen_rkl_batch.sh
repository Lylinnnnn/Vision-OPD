#!/bin/bash
# =============================================================================
# Batch run: Frozen Teacher experiments (RKL + JSD)
# 6 groups: pairwise student/teacher swap + JSD variants
# Sequential execution, merge+upload+cleanup after each experiment
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

LOG_DIR="res-opd/logs/frozen_batch_v4"
mkdir -p "$LOG_DIR"

OSS_BASE="oss://industry-algo/yanlin/ckpt/OPD"

# Note: Real-time ckpt upload is handled by a separate tmux session (ckpt-upload-watcher).
# This script only cleans up local ckpt dirs after each experiment finishes.
cleanup_local_ckpts() {
    local CKPT_DIR="$1"
    if [[ -d "$CKPT_DIR" ]]; then
        echo "  Cleaning up local ckpt: $CKPT_DIR"
        rm -rf "$CKPT_DIR"
    fi
}

echo "============================================================"
echo " Frozen Teacher Batch Experiments (v4)"
echo " Start time: $(date)"
echo "============================================================"

# --- Task 1: s224, t448, frozen+RKL ---
echo ""
echo "[1/6] s224-t448 frozen+RKL | $(date)"
STUDENT_PX=224 TARGET_PX=448 TEACHER_PX=0 \
    ALPHA=1.0 TEACHER_MODE=frozen \
    SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s224_t448_frozen_rkl.log"
cleanup_local_ckpts "${RES_OPD_ROOT}/checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s224-a1.0-frozen-e4"
echo "[1/6] DONE | $(date)"

# --- Task 2: s448, t224, frozen+RKL (swap of task 1) ---
echo ""
echo "[2/6] s448-t224 frozen+RKL | $(date)"
STUDENT_PX=448 TARGET_PX=448 TEACHER_PX=224 \
    ALPHA=1.0 TEACHER_MODE=frozen \
    SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s448_t224_frozen_rkl.log"
cleanup_local_ckpts "${RES_OPD_ROOT}/checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s448-t224-a1.0-frozen-e4"
echo "[2/6] DONE | $(date)"

# --- Task 3: s336, t448, frozen+RKL ---
echo ""
echo "[3/6] s336-t448 frozen+RKL | $(date)"
STUDENT_PX=336 TARGET_PX=448 TEACHER_PX=0 \
    ALPHA=1.0 TEACHER_MODE=frozen \
    SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s336_t448_frozen_rkl.log"
cleanup_local_ckpts "${RES_OPD_ROOT}/checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s336-a1.0-frozen-e4"
echo "[3/6] DONE | $(date)"

# --- Task 4: s448, t336, frozen+RKL (swap of task 3) ---
echo ""
echo "[4/6] s448-t336 frozen+RKL | $(date)"
STUDENT_PX=448 TARGET_PX=448 TEACHER_PX=336 \
    ALPHA=1.0 TEACHER_MODE=frozen \
    SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s448_t336_frozen_rkl.log"
cleanup_local_ckpts "${RES_OPD_ROOT}/checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s448-t336-a1.0-frozen-e4"
echo "[4/6] DONE | $(date)"

# --- Task 5: s224, t448, frozen+JSD ---
echo ""
echo "[5/6] s224-t448 frozen+JSD | $(date)"
STUDENT_PX=224 TARGET_PX=448 TEACHER_PX=0 \
    ALPHA=0.5 TEACHER_MODE=frozen \
    SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s224_t448_frozen_jsd.log"
cleanup_local_ckpts "${RES_OPD_ROOT}/checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s224-a0.5-frozen-e4"
echo "[5/6] DONE | $(date)"

# --- Task 6: s336, t448, frozen+JSD ---
echo ""
echo "[6/6] s336-t448 frozen+JSD | $(date)"
STUDENT_PX=336 TARGET_PX=448 TEACHER_PX=0 \
    ALPHA=0.5 TEACHER_MODE=frozen \
    SAVE_FREQ=20 TOTAL_EPOCHS=4 TEST_FREQ=20 \
    bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "$LOG_DIR/s336_t448_frozen_jsd.log"
cleanup_local_ckpts "${RES_OPD_ROOT}/checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s336-a0.5-frozen-e4"
echo "[6/6] DONE | $(date)"

echo ""
echo "============================================================"
echo " All 6 experiments completed at $(date)"
echo " Logs saved to: $LOG_DIR"
echo "============================================================"

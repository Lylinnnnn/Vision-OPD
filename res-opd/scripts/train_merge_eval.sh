#!/bin/bash
# =============================================================================
# Train → Merge → Eval Pipeline
#
# Runs training, then automatically merges the last checkpoint and evaluates.
# Eval results are stored in <ckpt_dir>/eval/.
#
# Usage:
#   bash res-opd/scripts/train_merge_eval.sh [extra env vars]
#
# Example:
#   STUDENT_PX=224 ALPHA=1.0 TEACHER_MODE=frozen \
#       bash res-opd/scripts/train_merge_eval.sh
#
# To skip eval (train + merge only):
#   SKIP_EVAL=1 STUDENT_PX=224 bash res-opd/scripts/train_merge_eval.sh
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

SKIP_EVAL="${SKIP_EVAL:-0}"
STUDENT_PX="${STUDENT_PX:-224}"

# --- Step 1: Train ---
echo "============================================================"
echo " [Pipeline] Step 1/3: Training"
echo " $(date)"
echo "============================================================"

TRAIN_LOG=$(mktemp)
bash "$SCRIPT_DIR/run_res_opd.sh" 2>&1 | tee "$TRAIN_LOG"

# Extract checkpoint dir from training log (avoids duplicating naming logic)
CKPT_DIR=$(grep "^Checkpoints:" "$TRAIN_LOG" | sed 's/^Checkpoints:[[:space:]]*//')
rm -f "$TRAIN_LOG"

if [[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]]; then
    echo "Error: Could not determine checkpoint directory from training output" >&2
    exit 1
fi

# Find the last global_step directory
LAST_STEP_DIR=$(ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
if [[ -z "$LAST_STEP_DIR" ]]; then
    echo "Error: No checkpoint found in $CKPT_DIR" >&2
    exit 1
fi
echo ""
echo "[Pipeline] Last checkpoint: $LAST_STEP_DIR"

# --- Step 2: Merge ---
echo ""
echo "============================================================"
echo " [Pipeline] Step 2/3: Merging checkpoint"
echo " $(date)"
echo "============================================================"

bash "$SCRIPT_DIR/merge_checkpoint.sh" "$LAST_STEP_DIR"

# --- Step 3: Eval (optional) ---
if [[ "$SKIP_EVAL" != "1" ]]; then
    echo ""
    echo "============================================================"
    echo " [Pipeline] Step 3/3: Evaluation"
    echo " $(date)"
    echo "============================================================"

    bash "$SCRIPT_DIR/eval_after_merge.sh" "$LAST_STEP_DIR" "$STUDENT_PX"
else
    echo ""
    echo "[Pipeline] Skipping evaluation (SKIP_EVAL=1)"
fi

echo ""
echo "============================================================"
echo " [Pipeline] Complete at $(date)"
echo " Checkpoint: $CKPT_DIR"
if [[ "$SKIP_EVAL" != "1" ]]; then
    EXPERIMENT_NAME="$(basename "$CKPT_DIR")"
    echo " Eval results: ${RES_OPD_ROOT}/eval_results/${EXPERIMENT_NAME}/"
fi
echo "============================================================"

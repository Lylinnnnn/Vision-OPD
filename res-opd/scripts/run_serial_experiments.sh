#!/bin/bash
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ⚠️  Log 规则: 串行训练日志写入 logs/<experiment_name>.log
#     每个子实验的日志由 run_res_opd.sh 的调用方通过 tee 写入。
#     禁止在 logs/ 下随意创建子目录！
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Serial Experiment Runner (Parameterized)
# Runs multiple training experiments sequentially on 8 GPUs.
# Each experiment automatically starts its own per-experiment ckpt_watcher.
#
# Usage:
#   bash scripts/run_serial_experiments.sh \
#       --experiments "STUDENT_PX=448 TEACHER_PX=0" "STUDENT_PX=448 TEACHER_PX=448" \
#       [--alpha 0.5] [--teacher-mode ema] [--epochs 2]
#
# Examples:
#   # Run t0 and t448 serially
#   bash scripts/run_serial_experiments.sh \
#       --experiments "STUDENT_PX=448 TEACHER_PX=0" "STUDENT_PX=448 TEACHER_PX=448"
#
#   # Run t224 and t336 with custom epochs
#   bash scripts/run_serial_experiments.sh \
#       --experiments "STUDENT_PX=448 TEACHER_PX=224" "STUDENT_PX=448 TEACHER_PX=336" \
#       --epochs 3
#
#   # Single experiment
#   bash scripts/run_serial_experiments.sh \
#       --experiments "STUDENT_PX=448 TEACHER_PX=200"
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

LOG_DIR="${RES_OPD_ROOT}/logs"
mkdir -p "$LOG_DIR"

# Defaults
ALPHA=0.5
TEACHER_MODE=ema
EPOCHS=2
EXPERIMENTS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiments)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXPERIMENTS+=("$1")
                shift
            done
            ;;
        --alpha)        ALPHA="$2"; shift 2 ;;
        --teacher-mode) TEACHER_MODE="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#EXPERIMENTS[@]} -eq 0 ]]; then
    echo "Error: --experiments is required" >&2
    echo "Usage: $0 --experiments \"STUDENT_PX=448 TEACHER_PX=0\" \"STUDENT_PX=448 TEACHER_PX=448\"" >&2
    exit 1
fi

TOTAL=${#EXPERIMENTS[@]}
echo "=========================================="
echo " Serial Experiment Runner"
echo " Experiments: ${TOTAL}"
echo " Alpha: ${ALPHA}, Mode: ${TEACHER_MODE}, Epochs: ${EPOCHS}"
echo " Started at: $(date)"
echo "=========================================="

for i in "${!EXPERIMENTS[@]}"; do
    idx=$((i + 1))
    env_vars="${EXPERIMENTS[$i]}"

    # Derive experiment name for log file (matches run_res_opd.sh naming)
    student_px=$(echo "$env_vars" | grep -oP 'STUDENT_PX=\K[0-9]+' || echo "448")
    teacher_px=$(echo "$env_vars" | grep -oP 'TEACHER_PX=\K[0-9]+' || echo "448")
    log_name="Res-OPD-Qwen3VL-2B-Instruct-s${student_px}-t${teacher_px}-a${ALPHA}-${TEACHER_MODE}-e${EPOCHS}"

    echo ""
    echo "[${idx}/${TOTAL}] Starting: ${env_vars}"
    echo "  Log: ${LOG_DIR}/${log_name}.log"

    # Use env command instead of eval for safety
    env ${env_vars} ALPHA=${ALPHA} TEACHER_MODE=${TEACHER_MODE} TOTAL_EPOCHS=${EPOCHS} \
        bash res-opd/scripts/run_res_opd.sh 2>&1 | tee "${LOG_DIR}/${log_name}.log"

    echo "[${idx}/${TOTAL}] Completed: ${env_vars}"
done

echo ""
echo "=========================================="
echo " All ${TOTAL} experiments completed!"
echo " Finished at: $(date)"
echo "=========================================="

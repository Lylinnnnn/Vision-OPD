#!/usr/bin/env bash
# =============================================================================
# OPD Eval Trace Scoring Launcher
#
# Runs forced scoring on eval_results.jsonl with automatic OSS checkpoint
# fetch, cuDNN fix, and optional cleanup.
#
# Usage:
#   # Score a single experiment (auto-detect paths from model name):
#   bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25
#
#   # Full options:
#   bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25 \
#       --max-samples 50 --topk 20
#
#   # Run in tmux:
#   tmux new-session -d -s opd_score_tr025 \
#       "cd /path/to/Vision-OPD && bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25 --max-samples 50 2>&1 | tee res-opd/logs/score_tr0.25.log"
#
# Supported experiment shortcuts:
#   sr1.0-tr0.25  sr1.0-tr0.5  sr1.0-tr0.75
#   sr0.25-tr1.0  sr0.5-tr1.0  sr0.75-tr1.0
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(dirname "$SCRIPT_DIR")"
VISION_OPD_ROOT="$(dirname "$RES_OPD_ROOT")"

# =============================================================================
# EXPERIMENT SHORTCUT MAPPING
# =============================================================================

declare -A EXP_MODEL_NAME=(
    ["sr1.0-tr0.25"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.25-a0.5-ema-e1"
    ["sr1.0-tr0.5"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.5-a0.5-ema-e1"
    ["sr1.0-tr0.75"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a0.5-ema-e1"
    ["sr0.25-tr1.0"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr0.25-tr1.0-a0.5-ema-e1"
    ["sr0.5-tr1.0"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr0.5-tr1.0-a0.5-ema-e1"
    ["sr0.75-tr1.0"]="Res-OPD-Qwen3VL-2B-Instruct-orig-sr0.75-tr1.0-a0.5-ema-e1"
)

declare -A EXP_OSS_PATH=(
    ["sr1.0-tr0.25"]="oss://industry-algo/yanlin/ckpt/OPD/v4/ResOPD_orig_sr1.0_tr0.25_a0.5_ema-e1/global_step_46"
    ["sr1.0-tr0.5"]="oss://industry-algo/yanlin/ckpt/OPD/v4/ResOPD_orig_sr1.0_tr0.5_a0.5_ema-e1/global_step_46"
    ["sr1.0-tr0.75"]="oss://industry-algo/yanlin/ckpt/OPD/v4/ResOPD_orig_sr1.0_tr0.75_a0.5_ema-e1/global_step_46"
    ["sr0.25-tr1.0"]="oss://industry-algo/yanlin/ckpt/OPD/v4/ResOPD_orig_sr0.25_tr1.0_a0.5_ema-e1/global_step_46"
    ["sr0.5-tr1.0"]="oss://industry-algo/yanlin/ckpt/OPD/v4/ResOPD_orig_sr0.5_tr1.0_a0.5_ema-e1/global_step_46"
    ["sr0.75-tr1.0"]="oss://industry-algo/yanlin/ckpt/OPD/v4/ResOPD_orig_sr0.75_tr1.0_a0.5_ema-e1/global_step_46"
)

declare -A EXP_STUDENT_RATIO=(
    ["sr1.0-tr0.25"]="1.0" ["sr1.0-tr0.5"]="1.0" ["sr1.0-tr0.75"]="1.0"
    ["sr0.25-tr1.0"]="0.25" ["sr0.5-tr1.0"]="0.5" ["sr0.75-tr1.0"]="0.75"
)

declare -A EXP_TEACHER_RATIO=(
    ["sr1.0-tr0.25"]="0.25" ["sr1.0-tr0.5"]="0.5" ["sr1.0-tr0.75"]="0.75"
    ["sr0.25-tr1.0"]="1.0" ["sr0.5-tr1.0"]="1.0" ["sr0.75-tr1.0"]="1.0"
)

declare -A EXP_CASE_ANALYSIS=(
    ["sr1.0-tr0.25"]="sr1.0-tr0.25_vs_baseline"
    ["sr1.0-tr0.5"]="sr1.0-tr0.5_vs_baseline"
    ["sr1.0-tr0.75"]="sr1.0-tr0.75_vs_baseline"
)

# =============================================================================
# PARSE ARGS
# =============================================================================

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment-shortcut> [extra args for score_opd_eval_trace.py]"
    echo ""
    echo "Available shortcuts:"
    for key in "${!EXP_MODEL_NAME[@]}"; do
        echo "  $key -> ${EXP_MODEL_NAME[$key]}"
    done
    exit 1
fi

EXP_KEY="$1"
shift

if [[ -z "${EXP_MODEL_NAME[$EXP_KEY]+x}" ]]; then
    echo "Error: Unknown experiment shortcut '$EXP_KEY'"
    echo "Available: ${!EXP_MODEL_NAME[*]}"
    exit 1
fi

MODEL_NAME="${EXP_MODEL_NAME[$EXP_KEY]}"
OSS_CHECKPOINT="${EXP_OSS_PATH[$EXP_KEY]}"
STUDENT_RATIO="${EXP_STUDENT_RATIO[$EXP_KEY]}"
TEACHER_RATIO="${EXP_TEACHER_RATIO[$EXP_KEY]}"
CASE_ANALYSIS_SUBDIR="${EXP_CASE_ANALYSIS[$EXP_KEY]:-}"
DATASET_TAG="${DATASET_TAG:-train1500_test300_original_sr1p0}"

# Derive paths
CHECKPOINT_STEP="global_step_46"
MODEL_PATH="${RES_OPD_ROOT}/checkpoints/${MODEL_NAME}/${CHECKPOINT_STEP}"

find_eval_results_dir() {
    local candidate
    local latest_root="${RES_OPD_ROOT}/eval_results/latest"

    # Current layout:
    #   eval_results/latest/{model_name}/{dataset}/eval_results.jsonl
    candidate="${latest_root}/${MODEL_NAME}/${DATASET_TAG}"
    if [[ -f "${candidate}/eval_results.jsonl" ]]; then
        echo "$candidate"
        return 0
    fi

    # Older layout:
    #   eval_results/latest/{model_name}_{step}/{dataset}/eval_results.jsonl
    candidate="${latest_root}/${MODEL_NAME}_${CHECKPOINT_STEP}/${DATASET_TAG}"
    if [[ -f "${candidate}/eval_results.jsonl" ]]; then
        echo "$candidate"
        return 0
    fi

    if [[ -d "$latest_root" ]]; then
        local found
        found="$(find "$latest_root" -mindepth 3 -maxdepth 3 -type f \
            -path "*/${DATASET_TAG}/eval_results.jsonl" \
            -path "*/${MODEL_NAME}/*" \
            | sort | head -n 1 || true)"
        if [[ -n "$found" ]]; then
            dirname "$found"
            return 0
        fi
    fi
    return 1
}

if ! EVAL_RESULTS_DIR="$(find_eval_results_dir)"; then
    EVAL_RESULTS_DIR="${RES_OPD_ROOT}/eval_results/latest/${MODEL_NAME}/${DATASET_TAG}"
fi
EVAL_RESULTS_JSONL="${EVAL_RESULTS_DIR}/eval_results.jsonl"

# Case analysis path (optional)
CASE_ANALYSIS_ARG=""
if [[ -n "$CASE_ANALYSIS_SUBDIR" ]]; then
    CASE_ANALYSIS_FILE="${EVAL_RESULTS_DIR}/case_analysis/all_cases_sorted.json"
    if [[ -f "$CASE_ANALYSIS_FILE" ]]; then
        CASE_ANALYSIS_ARG="--case-analysis $CASE_ANALYSIS_FILE"
    else
        # Fallback to v2 location
        CASE_ANALYSIS_FILE_V2="${RES_OPD_ROOT}/eval_results/v2/case_analysis/${CASE_ANALYSIS_SUBDIR}/all_cases_sorted.json"
        if [[ -f "$CASE_ANALYSIS_FILE_V2" ]]; then
            CASE_ANALYSIS_ARG="--case-analysis $CASE_ANALYSIS_FILE_V2"
        fi
    fi
fi

# =============================================================================
# ENVIRONMENT FIXES
# =============================================================================

# Fix cuDNN version mismatch
CONDA_CUDNN_LIB="$(python3 -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null || true)"
if [[ -n "$CONDA_CUDNN_LIB" && -d "$CONDA_CUDNN_LIB" ]]; then
    export LD_LIBRARY_PATH="${CONDA_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

# =============================================================================
# VALIDATE
# =============================================================================

if [[ ! -f "$EVAL_RESULTS_JSONL" ]]; then
    echo "Error: eval_results.jsonl not found at $EVAL_RESULTS_JSONL"
    exit 1
fi

# =============================================================================
# LAUNCH
# =============================================================================

echo "============================================================"
echo " OPD Eval Trace Scoring"
echo "============================================================"
echo "Experiment:       $EXP_KEY"
echo "Model:            $MODEL_NAME"
echo "Student ratio:    $STUDENT_RATIO"
echo "Teacher ratio:    $TEACHER_RATIO"
echo "Eval results:     $EVAL_RESULTS_JSONL"
echo "OSS checkpoint:   $OSS_CHECKPOINT"
echo "Local model path: $MODEL_PATH"
echo "Extra args:       $*"
echo "============================================================"

cd "$VISION_OPD_ROOT"

# shellcheck disable=SC2086
python -u res-opd/eval/score_opd_eval_trace.py \
    --model-path "$MODEL_PATH" \
    --oss-checkpoint "$OSS_CHECKPOINT" \
    --cleanup-after \
    --eval-results "$EVAL_RESULTS_JSONL" \
    --degradation-mode original \
    --student-ratio "$STUDENT_RATIO" \
    --teacher-ratio "$TEACHER_RATIO" \
    --topk 100 \
    --entropy \
    --max-samples 0 \
    --trace-scope eval \
    --overwrite \
    $CASE_ANALYSIS_ARG \
    "$@"

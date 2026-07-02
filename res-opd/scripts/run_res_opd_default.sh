#!/usr/bin/env bash
# =============================================================================
# Res-OPD Default Training Launcher
#
# Optimized defaults for 8x H20 GPUs with conservative settings.
# All parameters can still be overridden via environment variables.
#
# Usage:
#   bash res-opd/scripts/run_res_opd_default.sh
#
#   # Override specific params:
#   TRAIN_BATCH_SIZE=16 ROLLOUT_N=8 bash res-opd/scripts/run_res_opd_default.sh
#
#   # Run in tmux:
#   tmux new-session -d -s opd_train "cd /path/to/Vision-OPD && bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee res-opd/logs/train_default.log"
#
#   # Server example: frozen RKL, tr=0.75, keep 90% token mask (mask top 10% student-teacher delta)
#   cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD && tmux new-session -d -s opd_frozen_rkl_tr075_mask10 "EXPERIMENT_NAME=Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-mask10-e1 FORCE_FRESH_START=True TRAINER_RESUME_MODE=disable TOTAL_EPOCHS=1 DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 TEACHER_MODE=frozen ALPHA=1.0 OPD_TOKEN_MASK_PCT=0.10 OPD_TOKEN_MASK_METRIC=student_teacher_delta OPD_BUCKET_METRICS=True OPD_TRAIN_METRICS=True OPD_METRICS_ENTROPY=True OPD_TRACE_TOKEN=False OPD_MINI_EVAL_TRACE=True OPD_MINI_EVAL_TEST_FREQ=10 OPD_MINI_EVAL_MAX_SAMPLES=50 bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee res-opd/logs/Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-mask10-e1.log"
#
#   # Server example: frozen RKL, tr=0.75, recall-safer selective weighting
#   cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD && tmux new-session -d -s opd_frozen_rkl_tr075_sw "EXPERIMENT_NAME=Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw-e1 FORCE_FRESH_START=True TRAINER_RESUME_MODE=disable TOTAL_EPOCHS=1 DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 TEACHER_MODE=frozen ALPHA=1.0 OPD_SELECTIVE_WEIGHT=True OPD_SELECTIVE_WEIGHT_PROTECT=0.5 OPD_SELECTIVE_WEIGHT_UNCLEAR=0.5 OPD_SELECTIVE_WEIGHT_RISK=1.0 OPD_SELECTIVE_WEIGHT_OTHER=1.0 OPD_BUCKET_METRICS=True OPD_TRAIN_METRICS=True OPD_TRACE_TOKEN=False OPD_MINI_EVAL_TRACE=True OPD_MINI_EVAL_TEST_FREQ=10 OPD_MINI_EVAL_MAX_SAMPLES=50 bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee res-opd/logs/Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw-e1.log"
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(dirname "$SCRIPT_DIR")"

# =============================================================================
# OPTIMIZED DEFAULTS (conservative)
# =============================================================================

# --- Data loading ---
# 4 workers: better throughput for 10k dataset on JuiceFS
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-4}"

# --- Offload ---
# Disabled by default: H20 98GB has ample VRAM for Qwen3VL-2B actor+ref+rollout.
# Enable only if OOM occurs with large batch sizes.
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-False}"

# --- vLLM rollout ---
# 0.8: balanced KV cache utilization on H20 98GB
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"

# --- Sequence length ---
# Thinking models need room for reasoning traces; Instruct can still stop early.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-5120}"

# --- Logprob micro batch ---
# 4: faster logprob scoring while staying conservative for both Instruct and Thinking runs.
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export MAX_REPROMPT_LEN="${MAX_REPROMPT_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# --- Batch size ---
# 8: longer, comparable training curves for Instruct/Thinking full-5k runs.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export LR="${LR:-1e-6}"
export SAVE_FREQ="${SAVE_FREQ:-auto}"

# --- Mini eval ---
# Mini eval: 100 samples from val2017; frequency is resolved from actual steps/epoch.
export OPD_MINI_EVAL_TRACE="${OPD_MINI_EVAL_TRACE:-True}"
export OPD_MINI_EVAL_TEST_FREQ="${OPD_MINI_EVAL_TEST_FREQ:-auto}"
export OPD_MINI_EVAL_MAX_SAMPLES="${OPD_MINI_EVAL_MAX_SAMPLES:-100}"
export VAL_N="${VAL_N:-1}"
export VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
export VALIDATION_METRIC_MODE="${VALIDATION_METRIC_MODE:-mean_only}"

# --- Training metrics ---
export OPD_TRAIN_METRICS="${OPD_TRAIN_METRICS:-True}"
export OPD_METRICS_ENTROPY="${OPD_METRICS_ENTROPY:-True}"
export OPD_TRAIN_METRICS_VERBOSE="${OPD_TRAIN_METRICS_VERBOSE:-False}"
export OPD_SELECTIVE_METRICS_VERBOSE="${OPD_SELECTIVE_METRICS_VERBOSE:-False}"

# =============================================================================
# ENVIRONMENT FIXES
# =============================================================================

# Fix cuDNN version mismatch: prefer conda-bundled cuDNN over system /lib64.
# nvidia.cudnn.__file__ can be None in some installs, so use multiple fallbacks.
CONDA_CUDNN_LIB="$(python3 -c "
import os, sys
try:
    import nvidia.cudnn
    p = getattr(nvidia.cudnn, '__file__', None)
    if p:
        d = os.path.join(os.path.dirname(p), 'lib')
        if os.path.isdir(d):
            print(d); sys.exit(0)
except Exception:
    pass
try:
    import nvidia
    d = os.path.join(nvidia.__path__[0], 'cudnn', 'lib')
    if os.path.isdir(d):
        print(d); sys.exit(0)
except Exception:
    pass
" 2>/dev/null || true)"
if [[ -n "$CONDA_CUDNN_LIB" && -d "$CONDA_CUDNN_LIB" ]]; then
    export LD_LIBRARY_PATH="${CONDA_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

# =============================================================================
# LAUNCH
# =============================================================================

echo "============================================================"
echo " Res-OPD Default Training Launcher"
echo "============================================================"
echo "DATA_DATALOADER_NUM_WORKERS = $DATA_DATALOADER_NUM_WORKERS"
echo "ACTOR_PARAM_OFFLOAD         = $ACTOR_PARAM_OFFLOAD"
echo "ACTOR_OPTIMIZER_OFFLOAD     = $ACTOR_OPTIMIZER_OFFLOAD"
echo "REF_PARAM_OFFLOAD           = $REF_PARAM_OFFLOAD"
echo "ROLLOUT_GPU_MEMORY_UTIL     = $ROLLOUT_GPU_MEMORY_UTILIZATION"
echo "ROLLOUT_TENSOR_MP_SIZE      = $ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"
echo "MAX_PROMPT_LENGTH           = $MAX_PROMPT_LENGTH"
echo "MAX_RESPONSE_LENGTH         = $MAX_RESPONSE_LENGTH"
echo "ROLLOUT_MAX_BATCHED_TOKENS  = $ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "MAX_REPROMPT_LEN            = $MAX_REPROMPT_LEN"
echo "ROLLOUT_LOGPROB_MICRO_BSZ   = $ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
echo "REF_LOGPROB_MICRO_BSZ       = $REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
echo "TRAIN_BATCH_SIZE            = $TRAIN_BATCH_SIZE"
echo "PPO_MINI_BATCH_SIZE         = $PPO_MINI_BATCH_SIZE"
echo "LR                          = $LR"
echo "SAVE_FREQ                   = $SAVE_FREQ"
echo "OPD_MINI_EVAL_TRACE         = $OPD_MINI_EVAL_TRACE"
echo "OPD_MINI_EVAL_TEST_FREQ     = $OPD_MINI_EVAL_TEST_FREQ"
echo "OPD_MINI_EVAL_MAX_SAMPLES   = $OPD_MINI_EVAL_MAX_SAMPLES"
echo "VAL_N                       = $VAL_N"
echo "VAL_DO_SAMPLE               = $VAL_DO_SAMPLE"
echo "VALIDATION_METRIC_MODE      = $VALIDATION_METRIC_MODE"
echo "OPD_TRAIN_METRICS           = $OPD_TRAIN_METRICS"
echo "OPD_METRICS_ENTROPY         = $OPD_METRICS_ENTROPY"
echo "OPD_TRAIN_METRICS_VERBOSE   = $OPD_TRAIN_METRICS_VERBOSE"
echo "OPD_SELECTIVE_METRICS_VERBOSE = $OPD_SELECTIVE_METRICS_VERBOSE"
echo "OPD_SELECTIVE_WEIGHT        = ${OPD_SELECTIVE_WEIGHT:-False}"
echo "OPD_SELECTIVE_WEIGHT_MODE   = ${OPD_SELECTIVE_WEIGHT_MODE:-entropy_rkl_bucket}"
echo "============================================================"

exec bash "$SCRIPT_DIR/run_res_opd.sh" "$@"

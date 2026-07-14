#!/usr/bin/env bash
# =============================================================================
# Res-OPD Default Training Launcher
#
# Optimized defaults for current full5k tr0.75 frozen-RKL runs on 8x H20 GPUs.
# All parameters can still be overridden via environment variables.
#
# Usage:
#   bash res-opd/scripts/run_res_opd_default.sh
#
#   # Override specific params:
#   TEACHER_RATIO=1.0 TRAIN_BATCH_SIZE=32 bash res-opd/scripts/run_res_opd_default.sh
#
#   # Run in tmux:
#   tmux new-session -d -s opd_train "cd /path/to/Vision-OPD && bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee res-opd/logs/train_default.log"
#
#   # Current run examples; EXPERIMENT_NAME is generated from parameters.
#   bash res-opd/scripts/run_res_opd_default.sh
#   OPD_SELECTIVE_WEIGHT=True OPD_SELECTIVE_WEIGHT_MODE=risk_only_mask \
#     bash res-opd/scripts/run_res_opd_default.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/path_utils.sh"

# =============================================================================
# OPTIMIZED DEFAULTS (conservative)
# =============================================================================

DEFAULT_MODEL_ROOT="$(res_opd_default_model_root)"
DEFAULT_MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Qwen3VL-2B-Instruct}"
DEFAULT_MODEL_NAME_LC="$(basename "$DEFAULT_MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
if [[ "$DEFAULT_MODEL_NAME_LC" == *"8b"* ]]; then
    RES_OPD_MODEL_SIZE_PROFILE="8b"
    RES_OPD_DEFAULT_TRAIN_BATCH_SIZE=16
    RES_OPD_DEFAULT_PPO_MINI_BATCH_SIZE=16
    RES_OPD_DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
    RES_OPD_DEFAULT_LOGPROB_MICRO_BSZ=1
    RES_OPD_DEFAULT_ROLLOUT_BATCHED_FACTOR=2
    # 8B teacher forward can OOM with the historical non-thinking *4 budget,
    # but *1 is too tight for the current 8192+5120 protocol.  Use the same
    # conservative token budget as 8B Thinking.
    RES_OPD_DEFAULT_PPO_TOKEN_FACTOR=2
    RES_OPD_DEFAULT_ACTOR_PARAM_OFFLOAD=False
    RES_OPD_DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=True
    RES_OPD_DEFAULT_REF_PARAM_OFFLOAD=True
    RES_OPD_DEFAULT_ACTOR_CKPT_SAVE_CONTENTS="model,extra"
else
    RES_OPD_MODEL_SIZE_PROFILE="2b_or_smaller"
    RES_OPD_DEFAULT_TRAIN_BATCH_SIZE=32
    RES_OPD_DEFAULT_PPO_MINI_BATCH_SIZE=32
    RES_OPD_DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION=0.85
    RES_OPD_DEFAULT_LOGPROB_MICRO_BSZ=8
    RES_OPD_DEFAULT_ROLLOUT_BATCHED_FACTOR=8
    RES_OPD_DEFAULT_PPO_TOKEN_FACTOR=""
    RES_OPD_DEFAULT_ACTOR_PARAM_OFFLOAD=False
    RES_OPD_DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=False
    RES_OPD_DEFAULT_REF_PARAM_OFFLOAD=False
    RES_OPD_DEFAULT_ACTOR_CKPT_SAVE_CONTENTS="model,optimizer,extra"
fi

# --- Current experiment protocol ---
export DATASET_VERSION="${DATASET_VERSION:-full}"
export PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
export DEGRADATION_MODE="${DEGRADATION_MODE:-original}"
export STUDENT_RATIO="${STUDENT_RATIO:-1.0}"
export TEACHER_RATIO="${TEACHER_RATIO:-0.75}"
export TEACHER_MODE="${TEACHER_MODE:-frozen}"
export ALPHA="${ALPHA:-1.0}"

export TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
export FORCE_FRESH_START="${FORCE_FRESH_START:-True}"

# --- OSS cleanup ---
export POST_TRAIN_SYNC_TO_OSS="${POST_TRAIN_SYNC_TO_OSS:-True}"
export POST_TRAIN_CLEAN_LOCAL="${POST_TRAIN_CLEAN_LOCAL:-True}"
export POST_TRAIN_SYNC_ON_FAILURE="${POST_TRAIN_SYNC_ON_FAILURE:-True}"
export POST_TRAIN_CLEAN_LOGS="${POST_TRAIN_CLEAN_LOGS:-False}"
export AUTO_TEE_LOG="${AUTO_TEE_LOG:-True}"
export ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-$RES_OPD_DEFAULT_ACTOR_CKPT_SAVE_CONTENTS}"
export ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-$ACTOR_CKPT_SAVE_CONTENTS}"
export CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE="${CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE:-True}"
export CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD="${CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD:-False}"

# --- Data loading ---
# 4 workers: better throughput for 10k dataset on JuiceFS
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-4}"

# --- Offload ---
# Disabled by default: H20 98GB has ample VRAM for Qwen3VL-2B actor+ref+rollout.
# Enable only if OOM occurs with large batch sizes.
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-$RES_OPD_DEFAULT_ACTOR_PARAM_OFFLOAD}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-$RES_OPD_DEFAULT_ACTOR_OPTIMIZER_OFFLOAD}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-$RES_OPD_DEFAULT_REF_PARAM_OFFLOAD}"

# --- Training rollout ---
# 0.85: H20 98GB has enough headroom for Qwen3-VL 2B Instruct/Thinking.
# TP=1 is per rollout engine; training still uses TRAINER_N_GPUS_PER_NODE=8 by default.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-$RES_OPD_DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION}"
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"

# --- Sequence length ---
# Thinking models need room for reasoning traces; Instruct can still stop early.
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-5120}"

# --- Logprob micro batch ---
# 8: better H20 utilization for Qwen3-VL 2B; lower to 4 if long-thinking runs OOM.
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-$RES_OPD_DEFAULT_LOGPROB_MICRO_BSZ}"
export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-$RES_OPD_DEFAULT_LOGPROB_MICRO_BSZ}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * RES_OPD_DEFAULT_ROLLOUT_BATCHED_FACTOR))}"
if [[ -z "${PPO_MAX_TOKEN_LEN_PER_GPU+x}" ]]; then
    if [[ -n "$RES_OPD_DEFAULT_PPO_TOKEN_FACTOR" ]]; then
        export PPO_MAX_TOKEN_LEN_PER_GPU="$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * RES_OPD_DEFAULT_PPO_TOKEN_FACTOR))"
    elif [[ "$DEFAULT_MODEL_NAME_LC" == *"thinking"* ]]; then
        export PPO_MAX_TOKEN_LEN_PER_GPU="$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))"
    else
        export PPO_MAX_TOKEN_LEN_PER_GPU="$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 4))"
    fi
fi
export MAX_REPROMPT_LEN="${MAX_REPROMPT_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"

# --- Batch size ---
# 32 prompts/step keeps full-5k runs around 156 optimizer steps/epoch.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$RES_OPD_DEFAULT_TRAIN_BATCH_SIZE}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$RES_OPD_DEFAULT_PPO_MINI_BATCH_SIZE}"
export LR="${LR:-1e-6}"
export SAVE_FREQ="${SAVE_FREQ:-50}"

# --- Mini eval ---
# Keep long 8B reruns focused on training/checkpointing.  Set
# OPD_MINI_EVAL_TRACE=True explicitly when step-wise validation generations are
# needed.
export OPD_MINI_EVAL_TRACE="${OPD_MINI_EVAL_TRACE:-False}"
export OPD_MINI_EVAL_TEST_FREQ="${OPD_MINI_EVAL_TEST_FREQ:-500}"
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
CONDA_CUDNN_LIB="$("${PYTHON_BIN}" -c "
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
echo "MODEL_PATH                  = ${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
echo "MODEL_SIZE_PROFILE          = $RES_OPD_MODEL_SIZE_PROFILE"
echo "DATASET_VERSION             = $DATASET_VERSION"
echo "TOTAL_EPOCHS                = $TOTAL_EPOCHS"
echo "DEGRADATION_MODE            = $DEGRADATION_MODE"
echo "STUDENT_RATIO               = $STUDENT_RATIO"
echo "TEACHER_RATIO               = $TEACHER_RATIO"
echo "TEACHER_MODE                = $TEACHER_MODE"
echo "ALPHA                       = $ALPHA"
echo "TRAINER_RESUME_MODE         = $TRAINER_RESUME_MODE"
echo "FORCE_FRESH_START           = $FORCE_FRESH_START"
echo "POST_TRAIN_SYNC_TO_OSS      = $POST_TRAIN_SYNC_TO_OSS"
echo "POST_TRAIN_CLEAN_LOCAL      = $POST_TRAIN_CLEAN_LOCAL"
echo "POST_TRAIN_SYNC_ON_FAILURE  = $POST_TRAIN_SYNC_ON_FAILURE"
echo "POST_TRAIN_CLEAN_LOGS       = $POST_TRAIN_CLEAN_LOGS"
echo "ACTOR_CKPT_SAVE_CONTENTS    = $ACTOR_CKPT_SAVE_CONTENTS"
echo "ACTOR_CKPT_LOAD_CONTENTS    = $ACTOR_CKPT_LOAD_CONTENTS"
echo "CKPT_WATCHER_PRUNE_OPTIM    = $CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE"
echo "CKPT_WATCHER_KEEP_FSDP      = $CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD"
echo "AUTO_TEE_LOG                = $AUTO_TEE_LOG"
echo "DATA_DATALOADER_NUM_WORKERS = $DATA_DATALOADER_NUM_WORKERS"
echo "ACTOR_PARAM_OFFLOAD         = $ACTOR_PARAM_OFFLOAD"
echo "ACTOR_OPTIMIZER_OFFLOAD     = $ACTOR_OPTIMIZER_OFFLOAD"
echo "REF_PARAM_OFFLOAD           = $REF_PARAM_OFFLOAD"
echo "ROLLOUT_GPU_MEMORY_UTIL     = $ROLLOUT_GPU_MEMORY_UTILIZATION"
echo "ROLLOUT_TENSOR_MP_SIZE      = $ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE"
echo "MAX_PROMPT_LENGTH           = $MAX_PROMPT_LENGTH"
echo "MAX_RESPONSE_LENGTH         = $MAX_RESPONSE_LENGTH"
echo "ROLLOUT_MAX_BATCHED_TOKENS  = $ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "PPO_MAX_TOKEN_LEN_PER_GPU   = $PPO_MAX_TOKEN_LEN_PER_GPU"
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

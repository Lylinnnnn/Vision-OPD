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
#   TRAIN_BATCH_SIZE=64 ROLLOUT_N=8 bash res-opd/scripts/run_res_opd_default.sh
#
#   # Run in tmux:
#   tmux new-session -d -s opd_train "cd /path/to/Vision-OPD && bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee res-opd/logs/train_default.log"
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
# 0.92: maximize KV cache utilization on H20 98GB (offload keeps actor/ref safe)
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.92}"

# --- Logprob micro batch ---
# 4: higher GPU utilization for logprob computation with offload enabled
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"
export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-4}"

# --- Batch size ---
# 64: larger batch for 10k dataset, better GPU utilization
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-64}"

# --- Mini eval ---
# Enable mini eval trace by default; adjust freq/samples as needed
export OPD_MINI_EVAL_TRACE="${OPD_MINI_EVAL_TRACE:-True}"
export OPD_MINI_EVAL_TEST_FREQ="${OPD_MINI_EVAL_TEST_FREQ:-10}"
export OPD_MINI_EVAL_MAX_SAMPLES="${OPD_MINI_EVAL_MAX_SAMPLES:-50}"

# --- Training metrics ---
export OPD_TRAIN_METRICS="${OPD_TRAIN_METRICS:-True}"
export OPD_METRICS_ENTROPY="${OPD_METRICS_ENTROPY:-True}"

# =============================================================================
# ENVIRONMENT FIXES
# =============================================================================

# Fix cuDNN version mismatch: prefer conda-bundled cuDNN over system /lib64
CONDA_CUDNN_LIB="$(python3 -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'lib'))" 2>/dev/null || true)"
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
echo "ROLLOUT_LOGPROB_MICRO_BSZ   = $ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
echo "REF_LOGPROB_MICRO_BSZ       = $REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
echo "TRAIN_BATCH_SIZE            = $TRAIN_BATCH_SIZE"
echo "OPD_MINI_EVAL_TRACE         = $OPD_MINI_EVAL_TRACE"
echo "OPD_MINI_EVAL_TEST_FREQ     = $OPD_MINI_EVAL_TEST_FREQ"
echo "OPD_MINI_EVAL_MAX_SAMPLES   = $OPD_MINI_EVAL_MAX_SAMPLES"
echo "OPD_TRAIN_METRICS           = $OPD_TRAIN_METRICS"
echo "OPD_METRICS_ENTROPY         = $OPD_METRICS_ENTROPY"
echo "============================================================"

exec bash "$SCRIPT_DIR/run_res_opd.sh" "$@"

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
# 2 workers: safe starting point; increase to 4 if no OOM/blocking observed
export DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-2}"

# --- Offload ---
# Keep offload enabled by default for safety; disable when confirmed stable
export ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
export ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
export REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"

# --- vLLM rollout ---
# 0.85: better KV cache utilization without OOM risk (was 0.7)
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"

# --- Logprob micro batch ---
# 2: better GPU utilization for logprob computation (was 1)
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"

# --- Batch size ---
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"

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
echo "============================================================"

exec bash "$SCRIPT_DIR/run_res_opd.sh" "$@"

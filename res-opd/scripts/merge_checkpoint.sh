#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Merge FSDP-sharded checkpoint into a standard HuggingFace model
#
# Usage:
#   bash scripts/merge_checkpoint.sh <path_to_checkpoint>
#
# Example:
#   bash scripts/merge_checkpoint.sh \
#       ./checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s224-a0.5/global_step_50/
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"

DEFAULT_BASE_DIR="${RES_OPD_ROOT}/checkpoints"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"

if [ ! -d "${ACTOR_DIR}" ]; then
    echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
    echo "Expected structure: <checkpoint>/actor/" >&2
    exit 1
fi

echo "============================================================"
echo " Merging FSDP checkpoint"
echo "============================================================"
echo "Source:  ${ACTOR_DIR}"
echo "Target:  ${BASE_DIR}"

# Remove previously merged top-level files for clean overwrite
find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -delete

/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${ACTOR_DIR}" \
    --target_dir "${BASE_DIR}"

echo "Merge completed: ${BASE_DIR}"

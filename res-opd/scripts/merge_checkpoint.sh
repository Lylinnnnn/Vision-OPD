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
source "${SCRIPT_DIR}/path_utils.sh"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"

DEFAULT_BASE_DIR="${RES_OPD_ROOT}/checkpoints"
BASE_DIR="${BASE_DIR:-${1:-${DEFAULT_BASE_DIR}}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"
MERGE_SOURCE=""

if [[ -d "${ACTOR_DIR}" && -f "${ACTOR_DIR}/huggingface/config.json" ]]; then
    MERGE_SOURCE="${ACTOR_DIR}"
elif [[ -f "${BASE_DIR}/huggingface/config.json" ]]; then
    MERGE_SOURCE="${BASE_DIR}"
else
    echo "FSDP checkpoint directory not found under: ${BASE_DIR}" >&2
    echo "Expected either <checkpoint>/actor/ or a top-level SFT FSDP checkpoint." >&2
    exit 1
fi

echo "============================================================"
echo " Merging FSDP checkpoint"
echo "============================================================"
echo "Source:  ${MERGE_SOURCE}"
echo "Target:  ${BASE_DIR}"

# Remove previously merged top-level files for clean overwrite.
#
# Do not delete arbitrary top-level files here: FSDP checkpoints keep metadata
# such as data.pt in BASE_DIR, and the watcher uses marker files in the same
# directory. Only remove known HuggingFace merge outputs and partial leftovers.
find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f \( \
    -name "*.safetensors" -o \
    -name "*.safetensors.index.json" -o \
    -name "*.bin" -o \
    -name "*.bin.index.json" -o \
    -name "config.json" -o \
    -name "generation_config.json" -o \
    -name "preprocessor_config.json" -o \
    -name "processor_config.json" -o \
    -name "tokenizer.json" -o \
    -name "tokenizer_config.json" -o \
    -name "special_tokens_map.json" -o \
    -name "chat_template.jinja" -o \
    -name "merges.txt" -o \
    -name "vocab.json" \
\) -print -delete

PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"
"${PYTHON_BIN}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${MERGE_SOURCE}" \
    --target_dir "${BASE_DIR}"

echo "Merge completed: ${BASE_DIR}"

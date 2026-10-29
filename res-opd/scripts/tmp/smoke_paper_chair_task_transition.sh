#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
source "${RES_OPD_ROOT}/scripts/path_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
MODEL_ROOT="${MODEL_ROOT:-$(res_opd_default_model_root)}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-4}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8960}"
SMOKE_MAX_SAMPLES="${SMOKE_MAX_SAMPLES:-8}"
SMOKE_SUFFIX="${SMOKE_SUFFIX:-chair_official_transition_smoke_$(date +%Y%m%d_%H%M%S)_$$}"
SMOKE_INDICES="${SMOKE_INDICES:-2,4}"

cd "$VISION_OPD_ROOT"
"$PYTHON_BIN" res-opd/scripts/tmp/run_paper_chair_official_batch.py \
    --indices "$SMOKE_INDICES" \
    --base-port "$VLLM_BASE_PORT" \
    --gpu-list "$GPU_LIST" \
    --eval-shard-count "$EVAL_SHARD_COUNT" \
    --model-root "$MODEL_ROOT" \
    --output-suffix "$SMOKE_SUFFIX" \
    --max-samples "$SMOKE_MAX_SAMPLES" \
    --transition-wait-seconds 120 \
    --require-idle-gpus

echo "Smoke test passed: two OSS checkpoints ran sequentially, local checkpoints were cleaned, and port/GPU transitions were clean."

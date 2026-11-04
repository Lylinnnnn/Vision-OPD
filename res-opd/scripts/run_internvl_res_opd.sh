#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(dirname "$SCRIPT_DIR")"
source "${SCRIPT_DIR}/path_utils.sh"

: "${MODEL_PATH:?Set MODEL_PATH to the local InternVL3-2B-hf checkpoint}"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"
export PYTHON_BIN MODEL_PATH

case "${DATASET_VERSION:-full}" in
    full) TRAIN_PARQUET="${RES_OPD_ROOT}/data/train_5k.parquet" ;;
    quick) TRAIN_PARQUET="${RES_OPD_ROOT}/data/train_1.5k.parquet" ;;
    *) TRAIN_PARQUET="${RES_OPD_ROOT}/data/train.parquet" ;;
esac
"$PYTHON_BIN" "${RES_OPD_ROOT}/probes/preflight_training_images.py" "$TRAIN_PARQUET"
"$PYTHON_BIN" "${RES_OPD_ROOT}/probes/preflight_internvl3_hf.py" "$MODEL_PATH" \
    --teacher-ratio "${TEACHER_RATIO:-0.75}"

export MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-False}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
export ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-26624}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-26624}"
export ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-model,extra}"
export ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-$ACTOR_CKPT_SAVE_CONTENTS}"

exec bash "${SCRIPT_DIR}/run_res_opd_default.sh" "$@"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(dirname "$SCRIPT_DIR")"
source "$SCRIPT_DIR/path_utils.sh"
METHOD="${1:-}"
if [[ $# -ne 1 ]]; then
    echo "Usage: MODEL_PATH=/path/to/InternVL3-2B-hf bash $0 {frozen_rkl|lowres_sft|randommask|entropymask}" >&2
    exit 2
fi
case "$METHOD" in
    frozen_rkl|lowres_sft|randommask|entropymask) ;;
    *) echo "Unknown method: $METHOD" >&2; exit 2 ;;
esac

: "${MODEL_PATH:?Set MODEL_PATH to the InternVL3-2B-hf model directory}"
if [[ ! -f "$MODEL_PATH/config.json" ]]; then
    echo "Missing model config: $MODEL_PATH/config.json" >&2
    exit 1
fi
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"
export PYTHON_BIN
export RES_OPD_DATA_ROOT="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"

export DATASET_VERSION=full
export TOTAL_EPOCHS=1
export STUDENT_RATIO=1.0
export TEACHER_RATIO=0.5
export TEACHER_MODE=frozen
export DEGRADATION_MODE=original
export ALPHA=1.0
export SAVE_FREQ="${SAVE_FREQ:-50}"
export TRAINER_RESUME_MODE=disable
export FORCE_FRESH_START=True
export POST_TRAIN_SYNC_TO_OSS=True
export POST_TRAIN_CLEAN_LOCAL=True
export ACTOR_CKPT_SAVE_CONTENTS=model,extra
export ACTOR_CKPT_LOAD_CONTENTS=model,extra

if [[ "$METHOD" == lowres_sft ]]; then
    export SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-32}"
    export SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-1}"
    export SFT_USE_DYNAMIC_BSZ=False
    export SFT_USE_REMOVE_PADDING=False
    export ENGINE_PARAM_OFFLOAD=False
    export ENGINE_OPTIMIZER_OFFLOAD=False
    export SFT_CKPT_SAVE_CONTENTS=model,extra
    export SFT_CKPT_LOAD_CONTENTS=model,extra
    export STRICT_GENERATION_MODEL_CACHE=True
    export GEN_GPU_LIST="${GEN_GPU_LIST:-0}"
    "$PYTHON_BIN" "$RES_OPD_ROOT/probes/preflight_training_images.py" \
        "$RES_OPD_ROOT/data/train_5k.parquet"
    "$PYTHON_BIN" "$RES_OPD_ROOT/probes/preflight_internvl3_hf.py" \
        "$MODEL_PATH" --teacher-ratio 0.5
    "$PYTHON_BIN" "$RES_OPD_ROOT/probes/preflight_internvl_sft.py" \
        --model-path "$MODEL_PATH" --train-file "$RES_OPD_ROOT/data/train_5k.parquet"
    exec bash "$SCRIPT_DIR/run_lowres_sft.sh" --lowres-ratio 0.5 --train-batch-size "$SFT_TRAIN_BATCH_SIZE" \
        --micro-batch-size-per-gpu "$SFT_MICRO_BATCH_SIZE_PER_GPU"
fi

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
export TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-4}"
export ACTOR_USE_DYNAMIC_BSZ=False
export MODEL_USE_REMOVE_PADDING=False
export OPD_RISK_MASK_TOP_P=0.30
case "$METHOD" in
    frozen_rkl)
        export OPD_SELECTIVE_WEIGHT=False
        ;;
    randommask)
        export OPD_SELECTIVE_WEIGHT=True
        export OPD_SELECTIVE_WEIGHT_MODE=random_mask
        ;;
    entropymask)
        export OPD_SELECTIVE_WEIGHT=True
        export OPD_SELECTIVE_WEIGHT_MODE=entropy_mask
        ;;
esac

exec bash "$SCRIPT_DIR/run_internvl_res_opd.sh" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONDA_ENV="${CONDA_ENV:-rlsf-v-train}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
    if command -v conda >/dev/null 2>&1; then
        CONDA_BASE="$(conda info --base)"
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
        conda activate "$CONDA_ENV"
    else
        echo "Cannot find conda; set SKIP_CONDA_ACTIVATE=1 if the env is already active." >&2
        exit 1
    fi
fi

cd "$SCRIPT_DIR"

MODEL_SIZE="${MODEL_SIZE:-${1:-7b}}"

case "$MODEL_SIZE" in
    7b|llava-v1.5-7b|llava-7b)
        MODEL_ID="llava-v1.5-7b"
        DEFAULT_INPUT_MODEL="liuhaotian/llava-v1.5-7b"
        ;;
    13b|llava-v1.5-13b|llava-13b)
        MODEL_ID="llava-v1.5-13b"
        DEFAULT_INPUT_MODEL="liuhaotian/llava-v1.5-13b"
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE. Use 7b or 13b." >&2
        exit 1
        ;;
esac

DEFAULT_GENERATE_DIR="${SCRIPT_DIR}/generate/${MODEL_ID}"
DEFAULT_TRAINING_DATA_PATH="${DEFAULT_GENERATE_DIR}/${MODEL_ID}_dpo_data.json"
DEFAULT_RUN_NAME="${MODEL_ID}"

export INPUT_MODEL="${INPUT_MODEL:-$DEFAULT_INPUT_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GPU_LIST="${GPU_LIST:-$CUDA_VISIBLE_DEVICES}"
MASTER_PORT="${MASTER_PORT:-44564}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-accelerate_configs/zero2.json}"
TRAIN_RESULT_DIR="${TRAIN_RESULT_DIR:-${SCRIPT_DIR}/checkpoints}"

BETA="${BETA:-0.1}"
LEARNING_RATE="${LEARNING_RATE:-4e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
LORA_R="${LORA_R:-256}"
LORA_ALPHA="${LORA_ALPHA:-512}"
MM_PROJECTOR_LR="${MM_PROJECTOR_LR:-0}"
BITS="${BITS:-16}"
VISION_TOWER="${VISION_TOWER:-openai/clip-vit-large-patch14-336}"
MM_PROJECTOR_TYPE="${MM_PROJECTOR_TYPE:-mlp2x_gelu}"
MM_VISION_SELECT_LAYER="${MM_VISION_SELECT_LAYER:--2}"
IMAGE_ASPECT_RATIO="${IMAGE_ASPECT_RATIO:-pad}"
CONV_VERSION="${CONV_VERSION:-v1}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-2048}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DRY_RUN="${DRY_RUN:-0}"
SAVE_THROUGH_LOCAL_TMP="${SAVE_THROUGH_LOCAL_TMP:-1}"
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/tmp/rlsfv_llava_outputs}"

NUM_GPUS=$(echo "$GPU_LIST" | tr ',' '\n' | sed '/^$/d' | wc -l)
if [[ "$NUM_GPUS" -lt 1 ]]; then
    echo "No GPUs found in GPU_LIST/CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

if [[ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]]; then
    DENOM=$((NUM_GPUS * PER_DEVICE_BATCH_SIZE))
    if [[ "$DENOM" -eq 0 || $((GLOBAL_BATCH_SIZE % DENOM)) -ne 0 ]]; then
        echo "GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS * PER_DEVICE_BATCH_SIZE." >&2
        exit 1
    fi
    GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / DENOM))
fi

run_one() {
    local training_data_path="$1"
    local run_name="$2"
    local final_output_dir="${TRAIN_RESULT_DIR}/${run_name}"
    local run_output_dir="$final_output_dir"
    local local_output_dir=""

    if [[ ! -f "$training_data_path" ]]; then
        echo "Training data not found: $training_data_path" >&2
        exit 1
    fi

    mkdir -p "$TRAIN_RESULT_DIR"
    if [[ "$SAVE_THROUGH_LOCAL_TMP" == "1" ]]; then
        if [[ "$DRY_RUN" == "1" ]]; then
            local_output_dir="${LOCAL_OUTPUT_ROOT}/${run_name}.DRY_RUN"
        else
            mkdir -p "$LOCAL_OUTPUT_ROOT"
            local_output_dir="$(mktemp -d "${LOCAL_OUTPUT_ROOT}/${run_name}.XXXXXX")"
        fi
        run_output_dir="$local_output_dir"
    fi

    echo "======================================================================"
    echo "LLaVA training"
    echo "model_id: $MODEL_ID"
    echo "env: $CONDA_ENV"
    echo "model: $INPUT_MODEL"
    echo "data: $training_data_path"
    echo "gpus: $GPU_LIST"
    echo "global_batch_size: $GLOBAL_BATCH_SIZE"
    echo "per_device_batch_size: $PER_DEVICE_BATCH_SIZE"
    echo "gradient_accumulation_steps: $GRADIENT_ACCUMULATION_STEPS"
    echo "output: $final_output_dir"
    if [[ -n "$local_output_dir" ]]; then
        echo "local_output: $local_output_dir"
    fi
    echo "======================================================================"

    cmd=(
        deepspeed --include "localhost:${GPU_LIST}" --master_port "$MASTER_PORT" dpo_llava.py
        --deepspeed "$ACCELERATE_CONFIG"
        --model_name_or_path "$INPUT_MODEL"
        --output_dir "$run_output_dir"
        --run_name "$run_name"
        --train_data_path "$training_data_path"
        --beta "$BETA"
        --learning_rate "$LEARNING_RATE"
        --num_train_epochs "$NUM_TRAIN_EPOCHS"
        --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
        --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --warmup_ratio "$WARMUP_RATIO"
        --weight_decay "$WEIGHT_DECAY"
        --lr_scheduler_type "$LR_SCHEDULER_TYPE"
        --lora_enable True
        --lora_r "$LORA_R"
        --lora_alpha "$LORA_ALPHA"
        --mm_projector_lr "$MM_PROJECTOR_LR"
        --bits "$BITS"
        --vision_tower "$VISION_TOWER"
        --mm_projector_type "$MM_PROJECTOR_TYPE"
        --mm_vision_select_layer "$MM_VISION_SELECT_LAYER"
        --image_aspect_ratio "$IMAGE_ASPECT_RATIO"
        --mm_use_im_start_end False
        --mm_use_im_patch_token False
        --group_by_modality_length True
        --conv_version "$CONV_VERSION"
        --model_max_length "$MODEL_MAX_LENGTH"
        --bf16 True
        --tf32 True
        --gradient_checkpointing True
        --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
        --lazy_preprocess True
        --report_to none
        --eval_strategy no
        --save_strategy no
        --save_total_limit 1
        --logging_steps "$LOGGING_STEPS"
    )

    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY_RUN command:'
        printf ' %q' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
        if [[ -n "$local_output_dir" ]]; then
            echo "Copying saved adapter to $final_output_dir"
            mkdir -p "$final_output_dir"
            cp -a "$local_output_dir"/. "$final_output_dir"/
            rm -rf "$local_output_dir"
        fi
    fi
}

if [[ -n "${DATA_DIR:-}" ]]; then
    shopt -s nullglob
    files=("${DATA_DIR}"/*.json)
    if [[ "${#files[@]}" -eq 0 ]]; then
        echo "No .json files found in DATA_DIR: $DATA_DIR" >&2
        exit 1
    fi

    for file_path in "${files[@]}"; do
        filename="$(basename "$file_path")"
        run_name="${filename/_dpo_data.json/}"
        run_one "$file_path" "$run_name"
    done
else
    TRAINING_DATA_PATH="${TRAINING_DATA_PATH:-$DEFAULT_TRAINING_DATA_PATH}"
    RUN_NAME="${RUN_NAME:-$DEFAULT_RUN_NAME}"
    run_one "$TRAINING_DATA_PATH" "$RUN_NAME"
fi

echo "All requested LLaVA training jobs finished."

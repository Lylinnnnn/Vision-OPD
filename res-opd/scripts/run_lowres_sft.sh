#!/usr/bin/env bash
# =============================================================================
# Low-resolution sampled SFT for Res-OPD
#
# Pipeline:
#   1. Generate captions from low-resolution images with the base model.
#   2. Build a multi-turn SFT parquet that feeds original images and trains on
#      the low-res generated captions.
#   3. Run verl SFT.
#   4. Merge/upload checkpoints to OSS in the same layout eval_batch_from_oss.sh
#      can consume.
#
# Example:
#   bash res-opd/scripts/run_lowres_sft.sh --lowres-ratio 0.75
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
source "${SCRIPT_DIR}/path_utils.sh"
cd "$VISION_OPD_ROOT"

LOWRES_RATIO="${LOWRES_RATIO:-0.75}"
MAX_SAMPLES="${MAX_SAMPLES:--1}"
FORCE_REGENERATE="${FORCE_REGENERATE:-False}"
REUSE_SFT_DATA="${REUSE_SFT_DATA:-False}"
REUSE_COMPLETE_GENERATIONS="${REUSE_COMPLETE_GENERATIONS:-True}"
STRICT_GENERATION_MODEL_CACHE="${STRICT_GENERATION_MODEL_CACHE:-False}"
SKIP_GENERATION="${SKIP_GENERATION:-False}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --lowres-ratio|--teacher-ratio)
            LOWRES_RATIO="$2"; shift 2 ;;
        --model-path)
            MODEL_PATH="$2"; shift 2 ;;
        --experiment-name)
            EXPERIMENT_NAME="$2"; shift 2 ;;
        --max-samples)
            MAX_SAMPLES="$2"; shift 2 ;;
        --force-regenerate)
            FORCE_REGENERATE=True; shift ;;
        --reuse-sft-data)
            REUSE_SFT_DATA=True; shift ;;
        --skip-generation)
            SKIP_GENERATION=True; shift ;;
        --train-file)
            TASK_TRAIN_FILE="$2"; shift 2 ;;
        --save-freq)
            SAVE_FREQ="$2"; shift 2 ;;
        --total-epochs)
            TOTAL_EPOCHS="$2"; shift 2 ;;
        --total-training-steps)
            TRAINER_TOTAL_TRAINING_STEPS="$2"; shift 2 ;;
        --train-batch-size)
            SFT_TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --micro-batch-size-per-gpu)
            SFT_MICRO_BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
        --lr)
            LR="$2"; shift 2 ;;
        --vllm-port)
            VLLM_PORT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1 ;;
    esac
done

DEFAULT_MODEL_ROOT="$(res_opd_default_model_root)"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Qwen3VL-2B-Instruct}"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"

MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_NAME_LC="$(printf '%s' "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
if [[ "$MODEL_NAME_LC" == *"8b"* ]]; then
    DEFAULT_TRAIN_BATCH_SIZE=16
    DEFAULT_MICRO_BATCH_SIZE_PER_GPU=1
    DEFAULT_MAX_TOKEN_LEN_PER_GPU=26624
    DEFAULT_ENGINE_PARAM_OFFLOAD=False
    DEFAULT_ENGINE_OPTIMIZER_OFFLOAD=True
else
    DEFAULT_TRAIN_BATCH_SIZE=32
    DEFAULT_MICRO_BATCH_SIZE_PER_GPU=4
    DEFAULT_MAX_TOKEN_LEN_PER_GPU=26624
    DEFAULT_ENGINE_PARAM_OFFLOAD=False
    DEFAULT_ENGINE_OPTIMIZER_OFFLOAD=False
fi

DATASET_VERSION="${DATASET_VERSION:-full}"
if [[ "$DATASET_VERSION" == "full" ]]; then
    TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${RES_OPD_ROOT}/data/train_5k.parquet}"
    DATASET_TAG="full5k"
else
    TASK_TRAIN_FILE="${TASK_TRAIN_FILE:-${RES_OPD_ROOT}/data/train.parquet}"
    DATASET_TAG="legacy"
fi

SFT_TRAIN_BATCH_SIZE="${SFT_TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
SFT_MICRO_BATCH_SIZE_PER_GPU="${SFT_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_MICRO_BATCH_SIZE_PER_GPU}"
SFT_MAX_LENGTH="${SFT_MAX_LENGTH:-8192}"
SFT_MAX_TOKEN_LEN_PER_GPU="${SFT_MAX_TOKEN_LEN_PER_GPU:-$DEFAULT_MAX_TOKEN_LEN_PER_GPU}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:--1}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-8}"
LR="${LR:-1e-6}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"swanlab\"]}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"
FORCE_FRESH_START="${FORCE_FRESH_START:-False}"
POST_TRAIN_SYNC_TO_OSS="${POST_TRAIN_SYNC_TO_OSS:-True}"
POST_TRAIN_CLEAN_LOCAL="${POST_TRAIN_CLEAN_LOCAL:-True}"
POST_TRAIN_SYNC_ON_FAILURE="${POST_TRAIN_SYNC_ON_FAILURE:-True}"
POST_TRAIN_CLEAN_SFT_DATA="${POST_TRAIN_CLEAN_SFT_DATA:-False}"
SFT_CKPT_SAVE_CONTENTS="${SFT_CKPT_SAVE_CONTENTS:-model,extra}"
SFT_CKPT_LOAD_CONTENTS="${SFT_CKPT_LOAD_CONTENTS:-$SFT_CKPT_SAVE_CONTENTS}"
ENGINE_PARAM_OFFLOAD="${ENGINE_PARAM_OFFLOAD:-$DEFAULT_ENGINE_PARAM_OFFLOAD}"
ENGINE_OPTIMIZER_OFFLOAD="${ENGINE_OPTIMIZER_OFFLOAD:-$DEFAULT_ENGINE_OPTIMIZER_OFFLOAD}"
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
VLLM_PORT="${VLLM_PORT:-8320}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-${MODEL_NAME}}"
GEN_TENSOR_PARALLEL_SIZE="${GEN_TENSOR_PARALLEL_SIZE:-1}"
GEN_GPU_LIST="${GEN_GPU_LIST:-0}"
GEN_PARALLEL_WORKERS="${GEN_PARALLEL_WORKERS:-16}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-1024}"
GEN_MAX_MODEL_LEN="${GEN_MAX_MODEL_LEN:-8192}"
GEN_GPU_MEMORY_UTILIZATION="${GEN_GPU_MEMORY_UTILIZATION:-0.80}"
FINAL_ANSWER_ONLY="${FINAL_ANSWER_ONLY:-False}"
ENABLE_THINKING="${ENABLE_THINKING:-}"

format_ratio_tag() {
    "$PYTHON_BIN" - "$1" <<'PY' 2>/dev/null || echo "${1//./p}"
import sys
x = str(float(sys.argv[1])).rstrip("0").rstrip(".")
print(x)
PY
}

format_model_cache_tag() {
    local name="$1"
    local tag
    tag="$(printf '%s' "$name" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9._-]+/-/g; s/^-+//; s/-+$//')"
    if [[ -z "$tag" ]]; then
        tag="${name//\//-}"
    fi
    printf '%s\n' "$tag"
}

csv_to_hydra_list() {
    local csv="$1"
    local out="["
    local first=1
    local item
    IFS=',' read -ra items <<< "$csv"
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -n "$item" ]] || continue
        if [[ "$first" -eq 0 ]]; then
            out+=","
        fi
        out+="'$item'"
        first=0
    done
    out+="]"
    echo "$out"
}

get_oss_name() {
    local ckpt_dir_name="$1"
    local suffix
    if [[ "$ckpt_dir_name" == Res-OPD-Qwen3VL-2B-Instruct-* ]]; then
        suffix="${ckpt_dir_name#Res-OPD-Qwen3VL-2B-Instruct-}"
    elif [[ "$ckpt_dir_name" == Res-OPD-* ]]; then
        suffix="${ckpt_dir_name#Res-OPD-}"
    else
        suffix="$ckpt_dir_name"
    fi
    local epoch_tag=""
    if [[ "$suffix" =~ ^(.+)-(e[0-9]+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        epoch_tag="-${BASH_REMATCH[2]}"
    fi
    echo "ResOPD_${suffix//-/_}${epoch_tag}"
}

is_truthy() {
    case "${1:-}" in
        True|true|TRUE|1|yes|YES|y|Y) return 0 ;;
        *) return 1 ;;
    esac
}

safe_delete_dir() {
    local label="$1"
    local path="$2"
    local expected_prefix="$3"
    if [[ -z "$path" || "$path" != "$expected_prefix"* ]]; then
        echo "Refusing to delete ${label}: ${path}" >&2
        return 1
    fi
    rm -rf -- "$path"
}

hash_file() {
    local file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file"
    else
        shasum -a 256 "$file"
    fi
}

file_size_bytes() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

RATIO_TAG="$(format_ratio_tag "$LOWRES_RATIO")"
MODEL_CACHE_TAG="${MODEL_CACHE_TAG:-$(format_model_cache_tag "$MODEL_NAME")}"
if [[ -z "${EXPERIMENT_NAME:-}" ]]; then
    EXPERIMENT_NAME="Res-OPD-${MODEL_NAME}-orig-sr1.0-tr${RATIO_TAG}-sft-lowres-b${SFT_TRAIN_BATCH_SIZE}-${DATASET_TAG}-e${TOTAL_EPOCHS}"
fi

GENERATION_CACHE_DIR="${GENERATION_CACHE_DIR:-${RES_OPD_ROOT}/sft_data/${MODEL_CACHE_TAG}/tr${RATIO_TAG}}"
SFT_DATA_DIR="${SFT_DATA_DIR:-${RES_OPD_ROOT}/sft_data/${EXPERIMENT_NAME}}"
SFT_TRAIN_FILE="${SFT_TRAIN_FILE:-${SFT_DATA_DIR}/train.parquet}"
GENERATION_JSONL="${GENERATION_JSONL:-${GENERATION_CACHE_DIR}/lowres_generations.jsonl}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}}"
LOG_DIR="${RES_OPD_ROOT}/logs"
mkdir -p "$LOG_DIR" "$SFT_DATA_DIR" "$GENERATION_CACHE_DIR"

if ! [[ -d "$MODEL_PATH" ]]; then
    echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
    exit 1
fi
if ! [[ -f "$TASK_TRAIN_FILE" ]]; then
    echo "ERROR: train parquet does not exist: $TASK_TRAIN_FILE" >&2
    exit 1
fi

CONDA_CUDNN_LIB="$("${PYTHON_BIN}" -c "
import pathlib, site
for p in site.getsitepackages():
    d = pathlib.Path(p) / 'nvidia' / 'cudnn' / 'lib'
    if d.exists():
        print(d)
        break
" 2>/dev/null || true)"
if [[ -n "$CONDA_CUDNN_LIB" ]]; then
    export LD_LIBRARY_PATH="${CONDA_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

echo "============================================================"
echo " Res-OPD Low-res SFT"
echo "============================================================"
echo "Experiment:       $EXPERIMENT_NAME"
echo "Model:            $MODEL_PATH"
echo "Lowres ratio:     $LOWRES_RATIO"
echo "Cache model tag:  $MODEL_CACHE_TAG"
echo "Train parquet:    $TASK_TRAIN_FILE"
echo "SFT parquet:      $SFT_TRAIN_FILE"
echo "Generation cache: $GENERATION_CACHE_DIR"
echo "Generations:      $GENERATION_JSONL"
echo "Checkpoint dir:   $TRAINER_DEFAULT_LOCAL_DIR"
echo "OSS base:         $OSS_BASE"
echo "Batch/micro:      $SFT_TRAIN_BATCH_SIZE / $SFT_MICRO_BATCH_SIZE_PER_GPU"
echo "Total steps:      ${TRAINER_TOTAL_TRAINING_STEPS:-auto}"
echo "Save/test freq:   $SAVE_FREQ / $TRAINER_TEST_FREQ"
echo "Ckpt contents:    save=$SFT_CKPT_SAVE_CONTENTS load=$SFT_CKPT_LOAD_CONTENTS"

if is_truthy "$FORCE_FRESH_START"; then
    echo "FORCE_FRESH_START=True: cleaning local checkpoint dir"
    safe_delete_dir "checkpoint dir" "$TRAINER_DEFAULT_LOCAL_DIR" "${RES_OPD_ROOT}/checkpoints/"
    if is_truthy "$FORCE_REGENERATE"; then
        safe_delete_dir "sft data dir" "$SFT_DATA_DIR" "${RES_OPD_ROOT}/sft_data/"
        mkdir -p "$SFT_DATA_DIR"
    fi
fi

VLLM_PID=""
cleanup_vllm() {
    if [[ -n "${VLLM_PID:-}" ]]; then
        kill "$VLLM_PID" >/dev/null 2>&1 || true
        wait "$VLLM_PID" >/dev/null 2>&1 || true
        VLLM_PID=""
    fi
}
trap cleanup_vllm EXIT

wait_for_vllm() {
    local api_base="$1"
    local waited=0
    until curl -fsS "${api_base%/}/models" >/dev/null 2>&1; do
        sleep 2
        waited=$((waited + 2))
        if (( waited > 600 )); then
            echo "ERROR: vLLM server did not become ready after ${waited}s" >&2
            return 1
        fi
    done
}

build_prepare_args() {
    PREPARE_ARGS=(
        "${RES_OPD_ROOT}/pipelines/lowres_sft/prepare_data.py"
        --train-file "$TASK_TRAIN_FILE"
        --output-parquet "$SFT_TRAIN_FILE"
        --generation-jsonl "$GENERATION_JSONL"
        --model-name "$VLLM_MODEL_NAME"
        --lowres-ratio "$LOWRES_RATIO"
        --max-new-tokens "$GEN_MAX_NEW_TOKENS"
        --parallel-workers "$GEN_PARALLEL_WORKERS"
        --max-samples "$MAX_SAMPLES"
    )
    if is_truthy "$FORCE_REGENERATE"; then
        PREPARE_ARGS+=(--force-regenerate)
    fi
    if is_truthy "$FINAL_ANSWER_ONLY"; then
        PREPARE_ARGS+=(--final-answer-only)
    fi
    if [[ -n "$ENABLE_THINKING" ]]; then
        PREPARE_ARGS+=(--enable-thinking "$ENABLE_THINKING")
    fi
    if is_truthy "$STRICT_GENERATION_MODEL_CACHE"; then
        PREPARE_ARGS+=(--strict-model-cache)
    fi
}

if ! is_truthy "$SKIP_GENERATION"; then
    if [[ -f "$SFT_TRAIN_FILE" ]] && is_truthy "$REUSE_SFT_DATA"; then
        echo "Reusing existing SFT parquet: $SFT_TRAIN_FILE"
    else
        build_prepare_args
        GENERATION_CACHE_COMPLETE=False
        if ! is_truthy "$FORCE_REGENERATE" && is_truthy "$REUSE_COMPLETE_GENERATIONS"; then
            echo "Checking cached low-res generations before starting vLLM ..."
            if "$PYTHON_BIN" "${PREPARE_ARGS[@]}" --check-generations-only; then
                GENERATION_CACHE_COMPLETE=True
                echo "Cached low-res generations are complete; skipping vLLM generation."
            else
                echo "Cached low-res generations are incomplete or unmarked; vLLM generation is required."
            fi
        fi

        if is_truthy "$GENERATION_CACHE_COMPLETE"; then
            "$PYTHON_BIN" "${PREPARE_ARGS[@]}" --no-generate
        else
            echo "Starting vLLM for low-res label generation ..."
            CUDA_VISIBLE_DEVICES="$GEN_GPU_LIST" "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
                --model "$MODEL_PATH" \
                --served-model-name "$VLLM_MODEL_NAME" \
                --host "$VLLM_HOST" \
                --port "$VLLM_PORT" \
                --tensor-parallel-size "$GEN_TENSOR_PARALLEL_SIZE" \
                --max-model-len "$GEN_MAX_MODEL_LEN" \
                --gpu-memory-utilization "$GEN_GPU_MEMORY_UTILIZATION" \
                --trust-remote-code \
                > "${LOG_DIR}/vllm_${EXPERIMENT_NAME}.log" 2>&1 &
            VLLM_PID="$!"
            API_BASE="http://${VLLM_HOST}:${VLLM_PORT}/v1"
            wait_for_vllm "$API_BASE"

            "$PYTHON_BIN" "${PREPARE_ARGS[@]}" --api-base "$API_BASE"
            cleanup_vllm
        fi
    fi
else
    echo "SKIP_GENERATION=True: expecting existing SFT parquet at $SFT_TRAIN_FILE"
fi

if [[ ! -f "$SFT_TRAIN_FILE" ]]; then
    echo "ERROR: SFT parquet missing after generation step: $SFT_TRAIN_FILE" >&2
    exit 1
fi

SAVE_CONTENTS_HYDRA="$(csv_to_hydra_list "$SFT_CKPT_SAVE_CONTENTS")"
LOAD_CONTENTS_HYDRA="$(csv_to_hydra_list "$SFT_CKPT_LOAD_CONTENTS")"
TRAINING_STEP_OVERRIDE=()
if [[ -n "$TRAINER_TOTAL_TRAINING_STEPS" ]]; then
    TRAINING_STEP_OVERRIDE=(trainer.total_training_steps="$TRAINER_TOTAL_TRAINING_STEPS")
fi
TRAIN_EXIT_CODE=0
"$PYTHON_BIN" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$TRAINER_N_GPUS_PER_NODE" \
    -m verl.trainer.sft_trainer \
    --config-name sft_trainer_engine \
    model.path="$MODEL_PATH" \
    model.trust_remote_code=True \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    data.train_files="$SFT_TRAIN_FILE" \
    data.val_files=null \
    data.train_batch_size="$SFT_TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$SFT_MICRO_BATCH_SIZE_PER_GPU" \
    data.max_token_len_per_gpu="$SFT_MAX_TOKEN_LEN_PER_GPU" \
    data.max_length="$SFT_MAX_LENGTH" \
    data.truncation=right \
    data.pad_mode=no_padding \
    data.messages_key=messages \
    data.custom_cls.path="${RES_OPD_ROOT}/datasets/res_opd_sft_dataset.py" \
    data.custom_cls.name=ResOPDSFTDataset \
    data.ignore_input_ids_mismatch=True \
    engine.param_offload="$ENGINE_PARAM_OFFLOAD" \
    engine.optimizer_offload="$ENGINE_OPTIMIZER_OFFLOAD" \
    optim.lr="$LR" \
    optim.lr_warmup_steps_ratio="${LR_WARMUP_STEPS_RATIO:-0.1}" \
    optim.lr_scheduler_type="${LR_SCHEDULER_TYPE:-cosine}" \
    checkpoint.save_contents="$SAVE_CONTENTS_HYDRA" \
    checkpoint.load_contents="$LOAD_CONTENTS_HYDRA" \
    trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR" \
    trainer.project_name=Res-OPD-SFT \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    ${TRAINING_STEP_OVERRIDE[@]+"${TRAINING_STEP_OVERRIDE[@]}"} \
    trainer.save_freq="$SAVE_FREQ" \
    trainer.test_freq="$TRAINER_TEST_FREQ" \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.resume_mode="$TRAINER_RESUME_MODE" \
    trainer.n_gpus_per_node="$TRAINER_N_GPUS_PER_NODE" \
    trainer.max_ckpt_to_keep="${TRAINER_MAX_CKPT_TO_KEEP:-1}" \
    || TRAIN_EXIT_CODE=$?

upload_sft_checkpoints() {
    if ! is_truthy "$POST_TRAIN_SYNC_TO_OSS"; then
        return 0
    fi
    if ! command -v ossutil >/dev/null 2>&1; then
        echo "ERROR: POST_TRAIN_SYNC_TO_OSS=True requires ossutil in PATH." >&2
        return 1
    fi
    local oss_name
    oss_name="$(get_oss_name "$EXPERIMENT_NAME")"
    local step_dir step_name oss_step_path hf_dir model_bytes
    local uploaded_any=false
    shopt -s nullglob
    for step_dir in "${TRAINER_DEFAULT_LOCAL_DIR}"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        step_name="$(basename "$step_dir")"
        if [[ -f "${step_dir}/.oss_uploaded" ]]; then
            echo "Already uploaded: $step_name"
            uploaded_any=true
            continue
        fi
        if [[ ! -f "${step_dir}/huggingface/config.json" ]]; then
            echo "Skipping non-ready SFT checkpoint ${step_name}: missing huggingface/config.json" >&2
            continue
        fi
        if ! compgen -G "${step_dir}/model_world_size_*_rank_*.pt" >/dev/null; then
            echo "Skipping non-ready SFT checkpoint ${step_name}: missing model shards" >&2
            continue
        fi

        echo "Merging SFT checkpoint: ${step_name}"
        rm -f "${step_dir}"/model.safetensors "${step_dir}"/model.safetensors.sha256
        "$PYTHON_BIN" -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "$step_dir" \
            --target_dir "$step_dir"

        if [[ ! -s "${step_dir}/model.safetensors" ]]; then
            echo "ERROR: merge did not produce model.safetensors for ${step_name}" >&2
            return 1
        fi

        hash_file "${step_dir}/model.safetensors" > "${step_dir}/model.safetensors.sha256"
        model_bytes="$(file_size_bytes "${step_dir}/model.safetensors")"
        oss_step_path="${OSS_BASE}/${oss_name}/${step_name}"
        echo "Uploading ${step_name} to ${oss_step_path}/"
        ossutil cp "${step_dir}/model.safetensors" "${oss_step_path}/model.safetensors" -f
        ossutil cp "${step_dir}/model.safetensors.sha256" "${oss_step_path}/model.safetensors.sha256" -f
        hf_dir="${step_dir}/huggingface"
        for f in \
            config.json \
            tokenizer_config.json \
            tokenizer.json \
            chat_template.jinja \
            generation_config.json \
            processor_config.json \
            preprocessor_config.json \
            image_processor_config.json \
            video_processor_config.json \
            special_tokens_map.json \
            tokenizer.model \
            merges.txt \
            vocab.json; do
            if [[ -f "${hf_dir}/$f" ]]; then
                ossutil cp "${hf_dir}/$f" "${oss_step_path}/$f" -f
            fi
        done

        local oss_size
        oss_size="$(ossutil stat "${oss_step_path}/model.safetensors" 2>/dev/null | grep "Content-Length" | awk -F: '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
        if [[ "$oss_size" != "$model_bytes" ]]; then
            echo "ERROR: OSS upload size mismatch for ${step_name}: oss=${oss_size}, local=${model_bytes}" >&2
            return 1
        fi

        touch "${step_dir}/.oss_uploaded"
        uploaded_any=true
        rm -f "${step_dir}/model.safetensors" "${step_dir}/model.safetensors.sha256"
        find "$step_dir" -maxdepth 1 -type f \( \
            -name 'model_world_size_*_rank_*.pt' -o \
            -name 'optim_world_size_*_rank_*.pt' -o \
            -name 'extra_state_world_size_*_rank_*.pt' -o \
            -name 'data_*.pt' \
        \) -delete
    done
    shopt -u nullglob
    if [[ "$uploaded_any" != "true" ]]; then
        echo "ERROR: no SFT checkpoints were uploaded from ${TRAINER_DEFAULT_LOCAL_DIR}" >&2
        return 1
    fi
    if [[ -f "${SFT_DATA_DIR}/manifest.json" ]]; then
        ossutil cp -r "${SFT_DATA_DIR}/" "${OSS_BASE}/${oss_name}/training_artifacts/sft_run_data/" -f || true
    fi
    local generation_marker
    if [[ "$GENERATION_JSONL" == *.* ]]; then
        generation_marker="${GENERATION_JSONL%.*}.complete.json"
    else
        generation_marker="${GENERATION_JSONL}.complete.json"
    fi
    if [[ -f "$GENERATION_JSONL" && -f "$generation_marker" ]]; then
        ossutil cp "$GENERATION_JSONL" "${OSS_BASE}/${oss_name}/training_artifacts/lowres_generation_cache/lowres_generations.jsonl" -f || true
        ossutil cp "$generation_marker" "${OSS_BASE}/${oss_name}/training_artifacts/lowres_generation_cache/lowres_generations.complete.json" -f || true
    fi
}

POST_EXIT_CODE=0
if [[ "$TRAIN_EXIT_CODE" -eq 0 ]] || is_truthy "$POST_TRAIN_SYNC_ON_FAILURE"; then
    upload_sft_checkpoints || POST_EXIT_CODE=$?
fi

if [[ "$POST_EXIT_CODE" -eq 0 ]] && is_truthy "$POST_TRAIN_CLEAN_LOCAL"; then
    safe_delete_dir "checkpoint dir" "$TRAINER_DEFAULT_LOCAL_DIR" "${RES_OPD_ROOT}/checkpoints/"
fi
if [[ "$POST_EXIT_CODE" -eq 0 ]] && is_truthy "$POST_TRAIN_CLEAN_SFT_DATA"; then
    safe_delete_dir "sft data dir" "$SFT_DATA_DIR" "${RES_OPD_ROOT}/sft_data/"
fi

if [[ "$TRAIN_EXIT_CODE" -ne 0 ]]; then
    exit "$TRAIN_EXIT_CODE"
fi
exit "$POST_EXIT_CODE"

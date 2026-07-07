#!/usr/bin/env bash
# =============================================================================
# 2-step checkpoint upload smoke test.
#
# Runs a tiny Res-OPD training job, waits for checkpoint merge/upload, verifies
# local cleanup state, verifies OSS model.safetensors, then downloads the merged
# checkpoint through eval_batch_from_oss.sh and runs one CHAIR sample.
#
# This script is intentionally isolated under scripts/tmp and uses a unique
# experiment name by default.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
MODEL_PATH="${MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3-VL-8B-Thinking}"
MODEL_PROFILE="${MODEL_PROFILE:-qwen3vl_8b_thinking}"
MODEL_BASENAME="$(basename "$MODEL_PATH")"
MODEL_TAG="$(printf '%s' "$MODEL_BASENAME" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-')"
SMOKE_STAMP="${SMOKE_STAMP:-$(date '+%Y%m%d%H%M%S')}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-Res-OPD-smoke-${MODEL_TAG}-ckpt-upload-2step-${SMOKE_STAMP}}"
STEP="${SMOKE_STEP:-global_step_2}"
LOG_DIR="${RES_OPD_ROOT}/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/smoke_ckpt_upload_2step_${SMOKE_STAMP}.log}"
TRAIN_LOG="${LOG_DIR}/${EXPERIMENT_NAME}.log"
WATCHER_LOG="${LOG_DIR}/ckpt_watcher_${EXPERIMENT_NAME}.log"
CKPT_DIR="${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}"
STEP_DIR="${CKPT_DIR}/${STEP}"
RESULT_VERSION_TAG="${RESULT_VERSION_TAG:-smoke_ckpt_upload}"

mkdir -p "$LOG_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
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

count_files() {
    local path="$1"
    shift
    if [[ ! -e "$path" ]]; then
        echo 0
        return 0
    fi
    find "$path" -type f "$@" 2>/dev/null | wc -l | tr -d '[:space:]'
}

count_csv_entries() {
    local csv="${1:-}"
    csv="${csv//[[:space:]]/}"
    if [[ -z "$csv" ]]; then
        echo ""
        return 0
    fi
    echo "$(( $(awk -F, '{print NF}' <<< "$csv") ))"
}

infer_num_gpus() {
    if [[ -n "${SMOKE_NUM_GPUS:-}" ]]; then
        echo "$SMOKE_NUM_GPUS"
        return 0
    fi
    if [[ -n "${TRAINER_N_GPUS_PER_NODE:-}" ]]; then
        echo "$TRAINER_N_GPUS_PER_NODE"
        return 0
    fi
    if [[ -n "${GPU_LIST:-}" ]]; then
        count_csv_entries "$GPU_LIST"
        return 0
    fi
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]]; then
        count_csv_entries "$CUDA_VISIBLE_DEVICES"
        return 0
    fi
    echo 8
}

SMOKE_NUM_GPUS="$(infer_num_gpus)"
SMOKE_ROLLOUT_N="${ROLLOUT_N:-1}"
SMOKE_MIN_PPO_MINI_BATCH_SIZE="$(( (SMOKE_NUM_GPUS + SMOKE_ROLLOUT_N - 1) / SMOKE_ROLLOUT_N ))"
SMOKE_PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$SMOKE_MIN_PPO_MINI_BATCH_SIZE}"
SMOKE_TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$SMOKE_PPO_MINI_BATCH_SIZE}"
if (( SMOKE_PPO_MINI_BATCH_SIZE < SMOKE_MIN_PPO_MINI_BATCH_SIZE )); then
    log "WARNING: PPO_MINI_BATCH_SIZE=${SMOKE_PPO_MINI_BATCH_SIZE} is too small for ${SMOKE_NUM_GPUS} GPU(s) and rollout_n=${SMOKE_ROLLOUT_N}; using ${SMOKE_MIN_PPO_MINI_BATCH_SIZE}."
    SMOKE_PPO_MINI_BATCH_SIZE="$SMOKE_MIN_PPO_MINI_BATCH_SIZE"
fi
if (( SMOKE_TRAIN_BATCH_SIZE < SMOKE_PPO_MINI_BATCH_SIZE )); then
    log "WARNING: TRAIN_BATCH_SIZE=${SMOKE_TRAIN_BATCH_SIZE} is smaller than PPO_MINI_BATCH_SIZE=${SMOKE_PPO_MINI_BATCH_SIZE}; using ${SMOKE_PPO_MINI_BATCH_SIZE}."
    SMOKE_TRAIN_BATCH_SIZE="$SMOKE_PPO_MINI_BATCH_SIZE"
fi
SMOKE_NORMALIZED_PPO_MINI_BATCH_SIZE="$(( SMOKE_PPO_MINI_BATCH_SIZE * SMOKE_ROLLOUT_N / SMOKE_NUM_GPUS ))"

OSS_NAME="$(get_oss_name "$EXPERIMENT_NAME")"
OSS_STEP_PATH="${OSS_BASE%/}/${OSS_NAME}/${STEP}"

log "============================================================"
log "2-step checkpoint upload smoke test"
log "Experiment: ${EXPERIMENT_NAME}"
log "Model:      ${MODEL_PATH}"
log "OSS step:   ${OSS_STEP_PATH}"
log "GPUs:       ${SMOKE_NUM_GPUS}"
log "Batch:      train=${SMOKE_TRAIN_BATCH_SIZE}, ppo_mini=${SMOKE_PPO_MINI_BATCH_SIZE}, rollout_n=${SMOKE_ROLLOUT_N}, normalized_ppo_mini=${SMOKE_NORMALIZED_PPO_MINI_BATCH_SIZE}"
log "Log file:   ${LOG_FILE}"
log "============================================================"

log "[1/4] Running 3-step training and validating the step-2 periodic checkpoint ..."
env \
    MODEL_PATH="$MODEL_PATH" \
    MODEL_PROFILE="$MODEL_PROFILE" \
    EXPERIMENT_NAME="$EXPERIMENT_NAME" \
    FORCE_FRESH_START=True \
    TRAINER_RESUME_MODE=disable \
    TRAINER_N_GPUS_PER_NODE="$SMOKE_NUM_GPUS" \
    DATASET_VERSION="${DATASET_VERSION:-full}" \
    TOTAL_EPOCHS=1 \
    TRAINER_TOTAL_TRAINING_STEPS=3 \
    SAVE_FREQ=2 \
    SAVE_AT_EPOCH_END=False \
    TEST_AT_EPOCH_END=False \
    TEST_FREQ=-1 \
    OPD_MINI_EVAL_TRACE=False \
    TRAIN_BATCH_SIZE="$SMOKE_TRAIN_BATCH_SIZE" \
    PPO_MINI_BATCH_SIZE="$SMOKE_PPO_MINI_BATCH_SIZE" \
    ROLLOUT_N="$SMOKE_ROLLOUT_N" \
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}" \
    ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-1}" \
    POST_TRAIN_SYNC_TO_OSS=True \
    POST_TRAIN_SYNC_ON_FAILURE=True \
    POST_TRAIN_CLEAN_LOCAL=False \
    POST_TRAIN_UPLOAD_WAIT_SECONDS="${POST_TRAIN_UPLOAD_WAIT_SECONDS:-900}" \
    ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-model,extra}" \
    ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-model,extra}" \
    CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE=True \
    CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD=False \
    CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE=False \
    TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\"]}" \
    AUTO_TEE_LOG=True \
    bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee -a "$LOG_FILE"

log "[2/4] Verifying local checkpoint cleanup state ..."
if [[ ! -d "$STEP_DIR" ]]; then
    log "ERROR: expected checkpoint step dir not found: $STEP_DIR"
    exit 1
fi
if [[ ! -f "${STEP_DIR}/.oss_uploaded" ]]; then
    log "ERROR: missing upload marker: ${STEP_DIR}/.oss_uploaded"
    [[ -f "$WATCHER_LOG" ]] && tail -n 80 "$WATCHER_LOG" | tee -a "$LOG_FILE"
    exit 1
fi
if [[ -f "${STEP_DIR}/model.safetensors" ]]; then
    log "ERROR: merged model.safetensors should have been removed after upload: ${STEP_DIR}/model.safetensors"
    exit 1
fi

optim_count="$(count_files "$STEP_DIR" \( -name 'optim*.pt' -o -name '*optimizer*.pt' \))"
actor_model_count="$(count_files "${STEP_DIR}/actor" -name 'model_world_size_*_rank_*.pt')"
log "Local optimizer shard count: ${optim_count}"
log "Local actor model shard count after upload: ${actor_model_count}"
if [[ "$optim_count" != "0" ]]; then
    log "ERROR: optimizer shards still exist after watcher pruning."
    exit 1
fi
if [[ "$actor_model_count" != "0" ]]; then
    log "ERROR: actor model shards still exist after verified upload."
    exit 1
fi

log "[3/4] Verifying OSS merged checkpoint ..."
ossutil stat "${OSS_STEP_PATH}/model.safetensors" 2>&1 | tee -a "$LOG_FILE"
ossutil stat "${OSS_STEP_PATH}/model.safetensors.sha256" 2>&1 | tee -a "$LOG_FILE"

log "[4/4] Downloading from OSS and running one-sample CHAIR inference ..."
env \
    DATASET_VERSION="${DATASET_VERSION:-full}" \
    MODEL_PROFILE="$MODEL_PROFILE" \
    EVAL_BACKEND=single \
    EVAL_SHARDED=False \
    CLEANUP_LOCAL_CKPT=True \
    VLLM_BASE_PORT="${VLLM_BASE_PORT:-8877}" \
    VLLM_PORT_CLEANUP=True \
    CHAIR_MAX_SAMPLES=1 \
    CHAIR_MAX_NEW_TOKENS="${CHAIR_MAX_NEW_TOKENS:-128}" \
    CHAIR_PARALLEL_WORKERS=1 \
    RESULT_VERSION_TAG="$RESULT_VERSION_TAG" \
    VISION_ENABLE_THINKING="${VISION_ENABLE_THINKING:-True}" \
    bash res-opd/scripts/eval_batch_from_oss.sh \
        --oss-names "$OSS_NAME" \
        --local-names "$EXPERIMENT_NAME" \
        --step "$STEP" \
        --student-px 0 \
        --degradation-mode original \
        --student-ratio 1.0 \
        --version-tag "$RESULT_VERSION_TAG" \
        --eval-mode chair \
        --chair-max-samples 1 \
        --chair-max-new-tokens "${CHAIR_MAX_NEW_TOKENS:-128}" \
        --chair-parallel-workers 1 2>&1 | tee -a "$LOG_FILE"

RESULT_DIR="${RES_OPD_ROOT}/eval_results/${RESULT_VERSION_TAG}/full/${EXPERIMENT_NAME}_${STEP}/train5000_test1000_original_sr1p0"
if [[ ! -f "${RESULT_DIR}/eval_results.jsonl" ]]; then
    log "ERROR: expected CHAIR eval_results.jsonl not found: ${RESULT_DIR}/eval_results.jsonl"
    exit 1
fi
if [[ ! -s "${RESULT_DIR}/eval_results.jsonl" ]]; then
    log "ERROR: CHAIR eval_results.jsonl is empty: ${RESULT_DIR}/eval_results.jsonl"
    exit 1
fi

log "✅ Smoke test passed."
log "Train log:   ${TRAIN_LOG}"
log "Watcher log: ${WATCHER_LOG}"
log "Eval result: ${RESULT_DIR}/eval_results.jsonl"

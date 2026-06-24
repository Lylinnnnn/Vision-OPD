#!/bin/bash

set -eo pipefail

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ⚠️  IMPORTANT RULES:
#   1. 一次性/临时脚本必须放在 scripts/tmp/，禁止放在 scripts/ 根目录！
#      scripts/ 只保留常规最小运行脚本。用完的临时脚本请删除或归档。
#   2. Log 统一写入规则（禁止随意创建子目录）：
#      - 训练日志:  logs/<experiment_name>.log  (由调用方 tee 写入)
#      - Watcher:   logs/watcher_<experiment_name>.log  (watcher 自动写入)
#      - Eval:      无单独 log，结果在 eval_results/<version>/<exp>/<dataset>/
#      - 旧日志:    logs/archive/
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Res-OPD Training Script
# Resolution-Aware On-Policy Self-Distillation
#
# All Vision-OPD features preserved. Additional knobs for resolution distillation:
#   - STUDENT_PX:          student resolution (0 = no degradation)
#   - DEGRADATION_MODE:    square / original
#   - STUDENT_RATIO:       original-mode student degradation ratio
#   - TEACHER_RATIO:       original-mode teacher degradation ratio
#   - TARGET_PX:           target resolution for upsampling
#   - TEACHER_MODE:        ema / frozen / coevolving
#   - ALPHA:               loss interpolation (0.5=JSD, 1.0=RKL, 0.0=FKL)
#   - ROLLOUT_N:           number of rollouts per sample (1=pure KD, 8=GRPO+KD)
#   - OPD_SELECTIVE_VETO:  keep only top-p low-res veto tokens for distillation
#
# Examples:
#   # Default: student=224, teacher=EMA, loss=JSD
#   bash res-opd/scripts/run_res_opd.sh
#
#   # RKL loss, frozen teacher, 336px student
#   STUDENT_PX=336 ALPHA=1.0 TEACHER_MODE=frozen bash res-opd/scripts/run_res_opd.sh
#
#   # Co-evolving teacher (student=teacher same weights, different images)
#   TEACHER_MODE=coevolving bash res-opd/scripts/run_res_opd.sh
#
#   # Full Vision-OPD style: 8 rollouts with GRPO advantage
#   ROLLOUT_N=8 bash res-opd/scripts/run_res_opd.sh
#
#   # No degradation (student sees original images, only teacher signal)
#   STUDENT_PX=0 bash res-opd/scripts/run_res_opd.sh
#
#   # Original-size degradation: teacher sees 50% down/up sampled original image
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.5 bash res-opd/scripts/run_res_opd.sh
#
#   # Frozen RKL with selective low-res veto over the top 10% positive gaps
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 \
#     TEACHER_MODE=frozen ALPHA=1.0 OPD_SELECTIVE_VETO=True \
#     OPD_SELECTIVE_VETO_TOP_P=0.10 bash res-opd/scripts/run_res_opd.sh
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

CONFIG_NAME="res_opd"
MODEL_PATH="${MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}"

# --- Resolution params (online degradation) ---
STUDENT_PX="${STUDENT_PX:-224}"       # 0 = no degradation
TARGET_PX="${TARGET_PX:-448}"
TEACHER_PX="${TEACHER_PX:-448}"       # 0 = gray/blank image (no visual info), >=target_px = original image
DEGRADATION_MODE="${DEGRADATION_MODE:-square}"  # square / original
STUDENT_RATIO="${STUDENT_RATIO:-1.0}"           # original mode: 1.0 = original, 0.75/0.5/0.25 = down/up sample
TEACHER_RATIO="${TEACHER_RATIO:-1.0}"           # original mode: 0 = gray/blank, 1.0 = original

case "$DEGRADATION_MODE" in
    square|original)
        ;;
    *)
        echo "Error: Unknown DEGRADATION_MODE=$DEGRADATION_MODE (expected: square or original)" >&2
        exit 1
        ;;
esac

# --- Teacher mode ---
# Options: ema (default), frozen, coevolving
#   ema:         EMA teacher (teacher_model_source=legacy, teacher_regularization=ema)
#   frozen:      frozen base model as teacher (teacher_model_source=legacy, teacher_regularization=none)
#   coevolving:  actor IS the teacher (teacher_model_source=actor, teacher_regularization=none)
TEACHER_MODE="${TEACHER_MODE:-ema}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"

case "$TEACHER_MODE" in
    ema)
        TEACHER_MODEL_SOURCE="legacy"
        TEACHER_REGULARIZATION="ema"
        ;;
    frozen)
        TEACHER_MODEL_SOURCE="legacy"
        TEACHER_REGULARIZATION="none"
        ;;
    coevolving)
        TEACHER_MODEL_SOURCE="actor"
        TEACHER_REGULARIZATION="none"
        ;;
    *)
        echo "Error: Unknown TEACHER_MODE=$TEACHER_MODE (expected: ema, frozen, coevolving)" >&2
        exit 1
        ;;
esac

# --- Loss config ---
ALPHA="${ALPHA:-0.5}"                 # 0.5=JSD, 1.0=RKL, 0.0=FKL
LOSS_MODE="vopd"
DISTILLATION_TOPK="${DISTILLATION_TOPK:-100}"
OPD_SELECTIVE_VETO="${OPD_SELECTIVE_VETO:-False}"
OPD_SELECTIVE_VETO_TOP_P="${OPD_SELECTIVE_VETO_TOP_P:-0.10}"
OPD_SELECTIVE_VETO_MIN_SCORE="${OPD_SELECTIVE_VETO_MIN_SCORE:-0.0}"
OPD_SELECTIVE_VETO_MIN_TOKENS="${OPD_SELECTIVE_VETO_MIN_TOKENS:-1}"
OPD_SELECTIVE_VETO_NORMALIZE="${OPD_SELECTIVE_VETO_NORMALIZE:-True}"
OPD_TRAIN_METRICS="${OPD_TRAIN_METRICS:-True}"
OPD_METRICS_ENTROPY="${OPD_METRICS_ENTROPY:-False}"
OPD_TRACE_TOKEN="${OPD_TRACE_TOKEN:-False}"
OPD_TRACE_EVERY_N_STEPS="${OPD_TRACE_EVERY_N_STEPS:-5}"
OPD_TRACE_MAX_SAMPLES="${OPD_TRACE_MAX_SAMPLES:-4}"
OPD_TRACE_TOPK="${OPD_TRACE_TOPK:-$DISTILLATION_TOPK}"
OPD_TRACE_ENTROPY="${OPD_TRACE_ENTROPY:-True}"
OPD_TRACE_ONLY_FIRST_PPO_EPOCH="${OPD_TRACE_ONLY_FIRST_PPO_EPOCH:-True}"

# --- Training hyperparams ---
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
ROLLOUT_N="${ROLLOUT_N:-4}"           # 4 = multi-rollout KD (no GRPO loss)
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
LR="${LR:-2e-6}"
DONT_REPROMPT_ON_SELF_SUCCESS=True
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=1024
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
ACTOR_USE_DYNAMIC_BSZ=True
PPO_MAX_TOKEN_LEN_PER_GPU=$TRAIN_MAX_MODEL_LEN
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=1
ACTOR_PARAM_OFFLOAD=True
ACTOR_OPTIMIZER_OFFLOAD=True
REF_PARAM_OFFLOAD=True
TRAINER_N_GPUS_PER_NODE=8
TRAINER_NNODES="${WORLD_SIZE:-1}"
TRAINER_SAVE_FREQ="${SAVE_FREQ:-20}"
TRAINER_TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=1
TRAINER_LOGGER='["console","swanlab"]'
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"  # auto / disable / resume_path
FORCE_FRESH_START="${FORCE_FRESH_START:-False}"
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
POST_TRAIN_SYNC_TO_OSS="${POST_TRAIN_SYNC_TO_OSS:-False}"
POST_TRAIN_CLEAN_LOCAL="${POST_TRAIN_CLEAN_LOCAL:-False}"
POST_TRAIN_SYNC_ON_FAILURE="${POST_TRAIN_SYNC_ON_FAILURE:-False}"
POST_TRAIN_UPLOAD_WAIT_SECONDS="${POST_TRAIN_UPLOAD_WAIT_SECONDS:-1800}"
POST_TRAIN_UPLOAD_SWANLOG="${POST_TRAIN_UPLOAD_SWANLOG:-False}"
POST_TRAIN_CLEAN_LOGS="${POST_TRAIN_CLEAN_LOGS:-False}"
ROLLOUT_AGENT_NUM_WORKERS=8
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-0}"

case "$TRAINER_RESUME_MODE" in
    auto|disable|resume_path)
        ;;
    *)
        echo "Error: Unknown TRAINER_RESUME_MODE=$TRAINER_RESUME_MODE (expected: auto, disable, resume_path)" >&2
        exit 1
        ;;
esac

# --- Data paths ---
# Support both legacy (train.parquet / test.json) and full-scale (train_10k.parquet / test_1500.json) datasets.
# Set DATASET_VERSION=full to use the full-scale dataset; default is legacy.
DATA_DIR="${RES_OPD_ROOT}/data"
DATASET_VERSION="${DATASET_VERSION:-full}"
if [[ "$DATASET_VERSION" == "full" ]]; then
    TASK_TRAIN_FILE="${DATA_DIR}/train_5k.parquet"
    TASK_VAL_FILE="${DATA_DIR}/val.parquet"
else
    TASK_TRAIN_FILE="${DATA_DIR}/train.parquet"
    TASK_VAL_FILE="${DATA_DIR}/val.parquet"
fi
CUSTOM_DATASET_PATH="${RES_OPD_ROOT}/res_opd_dataset.py"

# --- Experiment naming ---
# Always include teacher_px in the name to avoid ambiguity (e.g. t0 vs t448)
MODEL_NAME=$(basename "$MODEL_PATH")
EPOCH_TAG="e${TRAINER_TOTAL_EPOCHS}"
if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
    : # Use externally provided EXPERIMENT_NAME
else
    if [[ "$DEGRADATION_MODE" == "original" ]]; then
        EXPERIMENT_NAME="Res-OPD-${MODEL_NAME}-orig-sr${STUDENT_RATIO}-tr${TEACHER_RATIO}-a${ALPHA}-${TEACHER_MODE}-${EPOCH_TAG}"
    else
        EXPERIMENT_NAME="Res-OPD-${MODEL_NAME}-s${STUDENT_PX}-t${TEACHER_PX}-a${ALPHA}-${TEACHER_MODE}-${EPOCH_TAG}"
    fi
fi
PROJECT_NAME="Res-OPD"
TRAINER_DEFAULT_LOCAL_DIR="${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAINER_ROLLOUT_DATA_DIR="${RES_OPD_ROOT}/rollouts/${EXPERIMENT_NAME}"
OPD_TRACE_DIR="${OPD_TRACE_DIR:-${RES_OPD_ROOT}/traces/${EXPERIMENT_NAME}}"
OPD_MINI_EVAL_TRACE="${OPD_MINI_EVAL_TRACE:-False}"
OPD_MINI_EVAL_GENERATION_DIR="${OPD_MINI_EVAL_GENERATION_DIR:-${RES_OPD_ROOT}/mini_eval_generations/${EXPERIMENT_NAME}}"
OPD_MINI_EVAL_MAX_SAMPLES="${OPD_MINI_EVAL_MAX_SAMPLES:-50}"
OPD_MINI_EVAL_TEST_FREQ="${OPD_MINI_EVAL_TEST_FREQ:-5}"
export EXPERIMENT="$EXPERIMENT_NAME"
WATCHER_SCRIPT="${RES_OPD_ROOT}/scripts/ckpt_upload_watcher.sh"
WATCHER_PID_FILE="${TRAINER_DEFAULT_LOCAL_DIR}/.watcher.pid"

is_truthy() {
    case "${1:-}" in
        True|true|TRUE|1|yes|YES|y|Y)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

if is_truthy "$OPD_MINI_EVAL_TRACE"; then
    TRAINER_VALIDATION_DATA_DIR="$OPD_MINI_EVAL_GENERATION_DIR"
    TRAINER_TEST_FREQ="${TEST_FREQ:-$OPD_MINI_EVAL_TEST_FREQ}"
    DATA_VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-$OPD_MINI_EVAL_MAX_SAMPLES}"
else
    TRAINER_VALIDATION_DATA_DIR="${TRAINER_VALIDATION_DATA_DIR:-null}"
    TRAINER_TEST_FREQ="${TEST_FREQ:-20}"
    DATA_VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
fi

delete_experiment_dir() {
    local label="$1"
    local path="$2"
    local expected_path="$3"

    if [[ -z "$path" || "$path" == "/" ]]; then
        echo "Error: refusing to delete invalid ${label} path: ${path:-<empty>}" >&2
        exit 1
    fi
    if [[ "$path" != "$expected_path" ]]; then
        echo "Error: refusing to delete unexpected ${label} path: $path" >&2
        echo "Expected: $expected_path" >&2
        exit 1
    fi
    if [[ -e "$path" ]]; then
        echo "  Removing ${label}: $path"
        rm -rf -- "$path"
    else
        echo "  ${label} does not exist: $path"
    fi
}

get_oss_name() {
    local ckpt_dir_name="$1"
    local suffix
    suffix="${ckpt_dir_name#Res-OPD-Qwen3VL-2B-Instruct-}"
    if [[ "$suffix" == "$ckpt_dir_name" ]]; then
        suffix="$ckpt_dir_name"
    fi

    local epoch_tag=""
    if [[ "$suffix" =~ ^(.+)-(e[0-9]+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        epoch_tag="-${BASH_REMATCH[2]}"
    fi

    echo "ResOPD_${suffix//-/_}${epoch_tag}"
}

all_checkpoint_steps_uploaded() {
    local step_dir
    for step_dir in "${TRAINER_DEFAULT_LOCAL_DIR}"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        if [[ ! -f "${step_dir}/.oss_uploaded" ]]; then
            return 1
        fi
    done
    return 0
}

wait_for_checkpoint_uploads() {
    local timeout_seconds="${1:-1800}"
    local interval_seconds=15
    local waited=0

    while true; do
        if all_checkpoint_steps_uploaded; then
            return 0
        fi
        if (( waited >= timeout_seconds )); then
            return 1
        fi
        echo "Waiting for checkpoint watcher uploads... (${waited}/${timeout_seconds}s)"
        sleep "$interval_seconds"
        waited=$((waited + interval_seconds))
    done
}

upload_file_to_oss() {
    local label="$1"
    local local_path="$2"
    local oss_path="$3"

    if [[ ! -f "$local_path" ]]; then
        echo "  ${label} not found, skip: $local_path"
        return 0
    fi
    echo "  Uploading ${label}: $local_path -> $oss_path"
    ossutil cp "$local_path" "$oss_path" -f
}

upload_dir_to_oss() {
    local label="$1"
    local local_path="$2"
    local oss_path="$3"

    if [[ ! -d "$local_path" ]]; then
        echo "  ${label} dir not found, skip: $local_path"
        return 0
    fi
    echo "  Uploading ${label}: $local_path -> $oss_path"
    ossutil cp -r "${local_path%/}/" "${oss_path%/}/" -f
}

stop_experiment_watcher() {
    if [[ -f "$WATCHER_PID_FILE" ]]; then
        local watcher_pid
        watcher_pid="$(cat "$WATCHER_PID_FILE" 2>/dev/null || true)"
        if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" 2>/dev/null; then
            echo "Stopping ckpt_watcher PID: $watcher_pid"
            kill "$watcher_pid" 2>/dev/null || true
            sleep 1
        fi
    fi
}

sync_training_artifacts_to_oss() {
    local train_exit_code="$1"

    if ! is_truthy "$POST_TRAIN_SYNC_TO_OSS"; then
        if is_truthy "$POST_TRAIN_CLEAN_LOCAL"; then
            echo "WARNING: POST_TRAIN_CLEAN_LOCAL=True ignored because POST_TRAIN_SYNC_TO_OSS is not enabled." >&2
        fi
        return 0
    fi
    if [[ "$train_exit_code" -ne 0 ]] && ! is_truthy "$POST_TRAIN_SYNC_ON_FAILURE"; then
        echo "Training failed with exit code ${train_exit_code}; skipping post-train OSS sync."
        return 0
    fi
    if ! command -v ossutil >/dev/null 2>&1; then
        echo "ERROR: POST_TRAIN_SYNC_TO_OSS=True requires ossutil in PATH." >&2
        return 1
    fi

    local oss_name
    oss_name="$(get_oss_name "$EXPERIMENT_NAME")"
    local oss_exp_path="${OSS_BASE%/}/${oss_name}"
    local oss_artifact_path="${oss_exp_path}/training_artifacts"

    echo "============================================================"
    echo " Post-train OSS sync"
    echo "============================================================"
    echo "OSS experiment:   $oss_exp_path"
    echo "Artifact target:  $oss_artifact_path"
    echo "Clean local:      $POST_TRAIN_CLEAN_LOCAL"

    if [[ -f "$WATCHER_SCRIPT" ]]; then
        if ! wait_for_checkpoint_uploads "$POST_TRAIN_UPLOAD_WAIT_SECONDS"; then
            echo "Checkpoint watcher did not finish before timeout; running one-shot upload scan."
            OSS_BASE="$OSS_BASE" bash "$WATCHER_SCRIPT" --watch-dir "$TRAINER_DEFAULT_LOCAL_DIR" --once
            wait_for_checkpoint_uploads 300
        fi
    else
        echo "WARNING: ckpt_upload_watcher.sh not found; checkpoint upload cannot be finalized." >&2
    fi

    local manifest_path="${TRAINER_DEFAULT_LOCAL_DIR}/post_train_artifacts_manifest.txt"
    mkdir -p "$TRAINER_DEFAULT_LOCAL_DIR"
    {
        echo "experiment_name=${EXPERIMENT_NAME}"
        echo "oss_experiment_path=${oss_exp_path}"
        echo "train_exit_code=${train_exit_code}"
        echo "synced_at=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "checkpoint_dir=${TRAINER_DEFAULT_LOCAL_DIR}"
        echo "rollout_dir=${TRAINER_ROLLOUT_DATA_DIR}"
        echo "trace_dir=${OPD_TRACE_DIR}"
        echo "mini_eval_generation_dir=${OPD_MINI_EVAL_GENERATION_DIR}"
        echo "mini_eval_trace_enabled=${OPD_MINI_EVAL_TRACE}"
    } > "$manifest_path"

    upload_file_to_oss "manifest" "$manifest_path" "${oss_artifact_path}/manifest.txt"
    upload_dir_to_oss "rollouts" "$TRAINER_ROLLOUT_DATA_DIR" "${oss_artifact_path}/rollouts"
    upload_dir_to_oss "mini-eval generations" \
        "$OPD_MINI_EVAL_GENERATION_DIR" \
        "${oss_artifact_path}/mini_eval_generations"
    upload_file_to_oss "trainer latest checkpoint marker" \
        "${TRAINER_DEFAULT_LOCAL_DIR}/latest_checkpointed_iteration.txt" \
        "${oss_artifact_path}/checkpoint_metadata/latest_checkpointed_iteration.txt"
    upload_file_to_oss "post-train manifest copy" \
        "$manifest_path" \
        "${oss_artifact_path}/checkpoint_metadata/post_train_artifacts_manifest.txt"

    upload_file_to_oss "training log" \
        "${RES_OPD_ROOT}/logs/${EXPERIMENT_NAME}.log" \
        "${oss_artifact_path}/logs/${EXPERIMENT_NAME}.log"
    upload_file_to_oss "watcher log" \
        "${RES_OPD_ROOT}/logs/ckpt_watcher_${EXPERIMENT_NAME}.log" \
        "${oss_artifact_path}/logs/ckpt_watcher_${EXPERIMENT_NAME}.log"
    upload_file_to_oss "legacy watcher log" \
        "${RES_OPD_ROOT}/logs/watcher_${EXPERIMENT_NAME}.log" \
        "${oss_artifact_path}/logs/watcher_${EXPERIMENT_NAME}.log"

    if is_truthy "$POST_TRAIN_UPLOAD_SWANLOG"; then
        upload_dir_to_oss "swanlab local logs" \
            "${VISION_OPD_ROOT}/swanlog" \
            "${oss_artifact_path}/swanlog"
    fi

    echo "Post-train OSS sync completed."

    if is_truthy "$POST_TRAIN_CLEAN_LOCAL"; then
        stop_experiment_watcher
        delete_experiment_dir "checkpoint dir" "$TRAINER_DEFAULT_LOCAL_DIR" "${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}"
        delete_experiment_dir "rollout dir" "$TRAINER_ROLLOUT_DATA_DIR" "${RES_OPD_ROOT}/rollouts/${EXPERIMENT_NAME}"
        if [[ "$OPD_MINI_EVAL_GENERATION_DIR" == "${RES_OPD_ROOT}/mini_eval_generations/${EXPERIMENT_NAME}" ]]; then
            delete_experiment_dir \
                "mini-eval generation dir" \
                "$OPD_MINI_EVAL_GENERATION_DIR" \
                "${RES_OPD_ROOT}/mini_eval_generations/${EXPERIMENT_NAME}"
        else
            echo "Skipping custom mini-eval generation dir cleanup: $OPD_MINI_EVAL_GENERATION_DIR"
        fi
        if is_truthy "$POST_TRAIN_CLEAN_LOGS"; then
            rm -f -- "${RES_OPD_ROOT}/logs/${EXPERIMENT_NAME}.log"
            rm -f -- "${RES_OPD_ROOT}/logs/ckpt_watcher_${EXPERIMENT_NAME}.log"
            rm -f -- "${RES_OPD_ROOT}/logs/watcher_${EXPERIMENT_NAME}.log"
        fi
    fi
}

if is_truthy "$FORCE_FRESH_START"; then
    echo "FORCE_FRESH_START=True: starting from scratch for ${EXPERIMENT_NAME}"
    TRAINER_RESUME_MODE="disable"
    if [[ -f "$WATCHER_PID_FILE" ]]; then
        watcher_pid="$(cat "$WATCHER_PID_FILE" 2>/dev/null || true)"
        if [[ -n "$watcher_pid" ]] && kill -0 "$watcher_pid" 2>/dev/null; then
            echo "  Stopping old ckpt_watcher PID: $watcher_pid"
            kill "$watcher_pid" 2>/dev/null || true
            sleep 1
        fi
    fi
    delete_experiment_dir "checkpoint dir" "$TRAINER_DEFAULT_LOCAL_DIR" "${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}"
    delete_experiment_dir "rollout dir" "$TRAINER_ROLLOUT_DATA_DIR" "${RES_OPD_ROOT}/rollouts/${EXPERIMENT_NAME}"
    if is_truthy "$OPD_MINI_EVAL_TRACE"; then
        if [[ "$OPD_MINI_EVAL_GENERATION_DIR" == "${RES_OPD_ROOT}/mini_eval_generations/${EXPERIMENT_NAME}" ]]; then
            delete_experiment_dir \
                "mini-eval generation dir" \
                "$OPD_MINI_EVAL_GENERATION_DIR" \
                "${RES_OPD_ROOT}/mini_eval_generations/${EXPERIMENT_NAME}"
        else
            echo "  Skipping custom mini-eval generation dir cleanup: $OPD_MINI_EVAL_GENERATION_DIR"
        fi
    fi
fi

mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"
if is_truthy "$OPD_MINI_EVAL_TRACE"; then
    mkdir -p "$OPD_MINI_EVAL_GENERATION_DIR"
fi

EXTRA_ARGS=("$@")

# =============================================================================
# ENVIRONMENT
# =============================================================================
export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONUNBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
# Disable cuDNN for conv ops to avoid libcudnn_graph.so.9 symbol mismatch (cudnnGetLibConfig)
export TORCH_CUDNN_V8_API_DISABLED=1
export CUDNN_FRONTEND_ATTN_DP_WORKSPACE_LIMIT=0
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
ulimit -c 0

# =============================================================================
# AUTO-START PER-EXPERIMENT CHECKPOINT UPLOAD WATCHER
# Each training run launches its own watcher that monitors ONLY this
# experiment's checkpoint directory, avoiding cross-machine conflicts
# on shared filesystems.
# =============================================================================
if [[ -f "$WATCHER_SCRIPT" ]]; then
    # Check if a watcher is already running for THIS experiment
    if [[ -f "$WATCHER_PID_FILE" ]] && kill -0 "$(cat "$WATCHER_PID_FILE")" 2>/dev/null; then
        echo "Per-experiment ckpt_watcher already running (PID: $(cat "$WATCHER_PID_FILE"))"
    else
        echo "Starting per-experiment ckpt_watcher for ${EXPERIMENT_NAME} ..."
        mkdir -p "$TRAINER_DEFAULT_LOCAL_DIR"
        OSS_BASE="$OSS_BASE" nohup bash "$WATCHER_SCRIPT" --watch-dir "$TRAINER_DEFAULT_LOCAL_DIR" > /dev/null 2>&1 &
        echo $! > "$WATCHER_PID_FILE"
        echo "  Watcher PID: $! (monitoring: $TRAINER_DEFAULT_LOCAL_DIR)"
    fi
else
    echo "WARNING: ckpt_upload_watcher.sh not found at $WATCHER_SCRIPT"
fi

# =============================================================================
# VALIDATION
# =============================================================================
if [[ ! -f "$TASK_TRAIN_FILE" ]]; then
    echo "Error: Training data not found at $TASK_TRAIN_FILE" >&2
    echo "Run: python res-opd/scripts/prepare_data.py --data-dir $DATA_DIR" >&2
    exit 1
fi
if [[ ! -f "$TASK_VAL_FILE" ]]; then
    echo "Error: Validation data not found at $TASK_VAL_FILE" >&2
    echo "Run: python res-opd/scripts/prepare_data.py --data-dir $DATA_DIR" >&2
    exit 1
fi
if [[ ! -f "$CUSTOM_DATASET_PATH" ]]; then
    echo "Error: Custom dataset not found at $CUSTOM_DATASET_PATH" >&2
    exit 1
fi

echo "============================================================"
echo " Res-OPD Training"
echo "============================================================"
echo "Model:            $MODEL_PATH"
echo "Degradation mode: $DEGRADATION_MODE"
echo "Student px:       $STUDENT_PX (0=no degradation)"
echo "Teacher px:       $TEACHER_PX (0=gray/blank visual input)"
echo "Target px:        $TARGET_PX"
echo "Student ratio:    $STUDENT_RATIO (original mode)"
echo "Teacher ratio:    $TEACHER_RATIO (original mode)"
echo "Teacher mode:     $TEACHER_MODE (src=$TEACHER_MODEL_SOURCE, reg=$TEACHER_REGULARIZATION, rate=$TEACHER_UPDATE_RATE)"
echo "Alpha (loss):     $ALPHA (0.5=JSD, 1.0=RKL, 0.0=FKL)"
echo "Selective veto:   $OPD_SELECTIVE_VETO (top_p=$OPD_SELECTIVE_VETO_TOP_P, min_score=$OPD_SELECTIVE_VETO_MIN_SCORE, normalize=$OPD_SELECTIVE_VETO_NORMALIZE)"
echo "Rollout N:        $ROLLOUT_N (1=pure KD, >1=GRPO+KD)"
echo "Learning rate:    $LR"
echo "Batch size:       $TRAIN_BATCH_SIZE"
echo "OPD metrics:      $OPD_TRAIN_METRICS (entropy curve=$OPD_METRICS_ENTROPY)"
echo "OPD token trace:  $OPD_TRACE_TOKEN (every ${OPD_TRACE_EVERY_N_STEPS} steps, max ${OPD_TRACE_MAX_SAMPLES}/rank, topk=${OPD_TRACE_TOPK})"
echo "Trace dir:        $OPD_TRACE_DIR"
echo "Mini-eval trace:  $OPD_MINI_EVAL_TRACE (test_freq=$TRAINER_TEST_FREQ, max_samples=$DATA_VAL_MAX_SAMPLES)"
echo "Mini-eval gen:    $TRAINER_VALIDATION_DATA_DIR"
echo "Resume mode:      $TRAINER_RESUME_MODE (force fresh=$FORCE_FRESH_START)"
echo "OSS base:         $OSS_BASE"
echo "Post-train OSS:   sync=$POST_TRAIN_SYNC_TO_OSS clean_local=$POST_TRAIN_CLEAN_LOCAL"
echo "Experiment:       $EXPERIMENT_NAME"
echo "Data:             $TASK_TRAIN_FILE"
echo "Dataset class:    ResOPDDataset ($CUSTOM_DATASET_PATH)"
echo "Checkpoints:      $TRAINER_DEFAULT_LOCAL_DIR"
echo "============================================================"

# =============================================================================
# LAUNCH TRAINING
# =============================================================================
PYTHON_BIN="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3"
set +e
"$PYTHON_BIN" -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[\"$TASK_VAL_FILE\"]" \
    data.val_batch_size=50 \
    data.val_max_samples=$DATA_VAL_MAX_SAMPLES \
    data.validation_shuffle=False \
    data.filter_overlong_prompts=False \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.trust_remote_code=True \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    data.custom_cls.path="$CUSTOM_DATASET_PATH" \
    data.custom_cls.name=ResOPDDataset \
    data.student_px=$STUDENT_PX \
    data.teacher_px=$TEACHER_PX \
    data.target_px=$TARGET_PX \
    data.degradation_mode="$DEGRADATION_MODE" \
    data.student_ratio=$STUDENT_RATIO \
    data.teacher_ratio=$TEACHER_RATIO \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=$LOSS_MODE \
    actor_rollout_ref.actor.calculate_entropy=$OPD_METRICS_ENTROPY \
    actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=10240 \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    actor_rollout_ref.actor.self_distillation.teacher_regularization=$TEACHER_REGULARIZATION \
    actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=hires_images \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.selective_veto_enabled=$OPD_SELECTIVE_VETO \
    actor_rollout_ref.actor.self_distillation.selective_veto_top_p=$OPD_SELECTIVE_VETO_TOP_P \
    actor_rollout_ref.actor.self_distillation.selective_veto_min_score=$OPD_SELECTIVE_VETO_MIN_SCORE \
    actor_rollout_ref.actor.self_distillation.selective_veto_min_tokens=$OPD_SELECTIVE_VETO_MIN_TOKENS \
    actor_rollout_ref.actor.self_distillation.selective_veto_normalize_by_selected=$OPD_SELECTIVE_VETO_NORMALIZE \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.train_metrics_enabled=$OPD_TRAIN_METRICS \
    actor_rollout_ref.actor.self_distillation.trace_enabled=$OPD_TRACE_TOKEN \
    actor_rollout_ref.actor.self_distillation.trace_dump_dir="$OPD_TRACE_DIR" \
    actor_rollout_ref.actor.self_distillation.trace_every_n_steps=$OPD_TRACE_EVERY_N_STEPS \
    actor_rollout_ref.actor.self_distillation.trace_max_samples=$OPD_TRACE_MAX_SAMPLES \
    actor_rollout_ref.actor.self_distillation.trace_topk=$OPD_TRACE_TOPK \
    actor_rollout_ref.actor.self_distillation.trace_entropy=$OPD_TRACE_ENTROPY \
    actor_rollout_ref.actor.self_distillation.trace_only_first_ppo_epoch=$OPD_TRACE_ONLY_FIRST_PPO_EPOCH \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False \
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    reward_model.enable=False \
    critic.model.path=$MODEL_PATH \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq="$TRAINER_TEST_FREQ" \
    trainer.validation_data_dir="$TRAINER_VALIDATION_DATA_DIR" \
    trainer.resume_mode=$TRAINER_RESUME_MODE \
    +trainer.save_at_epoch_end="${SAVE_AT_EPOCH_END:-True}" \
    +trainer.test_at_epoch_end="${TEST_AT_EPOCH_END:-False}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-6}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${EXTRA_ARGS[@]}"
TRAIN_EXIT_CODE=$?
set -e

sync_training_artifacts_to_oss "$TRAIN_EXIT_CODE"
exit "$TRAIN_EXIT_CODE"

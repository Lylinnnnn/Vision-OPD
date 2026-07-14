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
#   - DEGRADATION_MODE:    original
#   - STUDENT_RATIO:       student original-ratio degradation
#   - TEACHER_RATIO:       teacher original-ratio degradation
#   - STUDENT_PX/TEACHER_PX/TARGET_PX: legacy metadata/CLI compatibility only
#   - TEACHER_MODE:        ema / frozen / fixed / coevolving
#   - ALPHA:               loss interpolation (0.5=JSD, 1.0=RKL, 0.0=FKL)
#   - ROLLOUT_N:           number of rollouts per sample (1=pure KD, 8=GRPO+KD)
#   - OPD_SELECTIVE_VETO:  keep only top-p low-res veto tokens for distillation
#   - OPD_TOKEN_MASK_PCT:  mask highest-divergence token fraction per sample
#   - OPD_TOKEN_MASK_METRIC: loss / student_teacher_delta
#   - OPD_SELECTIVE_WEIGHT: soft-weight RKL/JSD tokens by entropy/loss buckets
#
# Examples:
#   # Co-evolving teacher (student=teacher same weights, different images)
#   TEACHER_MODE=coevolving bash res-opd/scripts/run_res_opd.sh
#
#   # Full Vision-OPD style: 8 rollouts with GRPO advantage
#   ROLLOUT_N=8 bash res-opd/scripts/run_res_opd.sh
#
#   # Original-size degradation: teacher sees 50% down/up sampled original image
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.5 bash res-opd/scripts/run_res_opd.sh
#
#   # Frozen RKL with selective low-res veto over the top 10% positive gaps
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 \
#     TEACHER_MODE=frozen ALPHA=1.0 OPD_SELECTIVE_VETO=True \
#     OPD_SELECTIVE_VETO_TOP_P=0.10 bash res-opd/scripts/run_res_opd.sh
#
#   # Keep 90% token-mask variant: mask top 10% highest low-res disagreement tokens
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 \
#     TEACHER_MODE=frozen ALPHA=1.0 OPD_TOKEN_MASK_PCT=0.10 \
#     OPD_TOKEN_MASK_METRIC=student_teacher_delta OPD_BUCKET_METRICS=True \
#     bash res-opd/scripts/run_res_opd.sh
#
#   # Keep 70% stricter variant: mask top 30%
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 \
#     TEACHER_MODE=frozen ALPHA=1.0 OPD_TOKEN_MASK_PCT=0.30 \
#     OPD_TOKEN_MASK_METRIC=student_teacher_delta OPD_BUCKET_METRICS=True \
#     bash res-opd/scripts/run_res_opd.sh
#
#   # Recall-safer weighted RKL: weaken low-risk/unclear tokens, keep risk tokens
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 \
#     TEACHER_MODE=frozen ALPHA=1.0 OPD_SELECTIVE_WEIGHT=True \
#     OPD_SELECTIVE_WEIGHT_PROTECT=0.5 OPD_SELECTIVE_WEIGHT_UNCLEAR=0.5 \
#     OPD_SELECTIVE_WEIGHT_RISK=1.0 OPD_SELECTIVE_WEIGHT_OTHER=1.0 \
#     bash res-opd/scripts/run_res_opd.sh
#
#   # Risk-only mask: keep low-res RKL only for top-p high-RKL x high-uncertainty tokens.
#   DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.75 \
#     TEACHER_MODE=frozen ALPHA=1.0 OPD_SELECTIVE_WEIGHT=True \
#     OPD_SELECTIVE_WEIGHT_MODE=risk_only_mask OPD_RISK_MASK_TOP_P=0.30 \
#     bash res-opd/scripts/run_res_opd.sh
# =============================================================================

# =============================================================================
# CONFIGURATION
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
source "${SCRIPT_DIR}/path_utils.sh"

CONFIG_NAME="res_opd"
DEFAULT_MODEL_ROOT="$(res_opd_default_model_root)"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Qwen3VL-2B-Instruct}"
MODEL_NAME=$(basename "$MODEL_PATH")
MODEL_NAME_LC="$(echo "$MODEL_NAME" | tr '[:upper:]' '[:lower:]')"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"
if [[ "$MODEL_NAME_LC" == *"8b"* ]]; then
    MODEL_SIZE_PROFILE="${MODEL_SIZE_PROFILE:-8b}"
    DEFAULT_TRAIN_BATCH_SIZE=16
    DEFAULT_PPO_MINI_BATCH_SIZE=16
    DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION=0.65
    DEFAULT_LOGPROB_MICRO_BSZ=1
    DEFAULT_ROLLOUT_BATCHED_FACTOR=2
    # Match the 8B Thinking-safe token budget by default.
    DEFAULT_PPO_TOKEN_FACTOR=2
    DEFAULT_ACTOR_PARAM_OFFLOAD=False
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=True
    DEFAULT_REF_PARAM_OFFLOAD=True
    DEFAULT_ACTOR_CKPT_SAVE_CONTENTS="model,extra"
    DEFAULT_MAX_ACTOR_CKPT_TO_KEEP=2
else
    MODEL_SIZE_PROFILE="${MODEL_SIZE_PROFILE:-2b_or_smaller}"
    DEFAULT_TRAIN_BATCH_SIZE=32
    DEFAULT_PPO_MINI_BATCH_SIZE=32
    DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION=0.85
    DEFAULT_LOGPROB_MICRO_BSZ=8
    DEFAULT_ROLLOUT_BATCHED_FACTOR=8
    DEFAULT_PPO_TOKEN_FACTOR=""
    DEFAULT_ACTOR_PARAM_OFFLOAD=False
    DEFAULT_ACTOR_OPTIMIZER_OFFLOAD=False
    DEFAULT_REF_PARAM_OFFLOAD=False
    DEFAULT_ACTOR_CKPT_SAVE_CONTENTS="model,optimizer,extra"
    DEFAULT_MAX_ACTOR_CKPT_TO_KEEP=1
fi

# --- Original-ratio params (online degradation) ---
STUDENT_PX="${STUDENT_PX:-0}"         # legacy metadata only
TARGET_PX="${TARGET_PX:-448}"         # legacy metadata only
TEACHER_PX="${TEACHER_PX:-448}"       # legacy metadata only
DEGRADATION_MODE="${DEGRADATION_MODE:-original}"
STUDENT_RATIO="${STUDENT_RATIO:-1.0}"           # 1.0 = original, 0.75/0.5/0.25 = down/up sample
TEACHER_RATIO="${TEACHER_RATIO:-1.0}"           # 0 = gray/blank, 1.0 = original

case "$DEGRADATION_MODE" in
    original)
        ;;
    *)
        echo "Error: Unknown DEGRADATION_MODE=$DEGRADATION_MODE (expected: original)" >&2
        exit 1
        ;;
esac

# --- Teacher mode ---
# Options: ema (default), frozen, fixed, coevolving
#   ema:         EMA teacher (teacher_model_source=legacy, teacher_regularization=ema)
#   frozen:      frozen base model as teacher (teacher_model_source=legacy, teacher_regularization=none)
#   fixed:       frozen external teacher model from TEACHER_MODEL_PATH
#   coevolving:  actor IS the teacher (teacher_model_source=actor, teacher_regularization=none)
TEACHER_MODE="${TEACHER_MODE:-ema}"
TEACHER_UPDATE_RATE="${TEACHER_UPDATE_RATE:-0.05}"
TEACHER_MODEL_PATH="${TEACHER_MODEL_PATH:-}"

case "$TEACHER_MODE" in
    ema)
        TEACHER_MODEL_SOURCE="legacy"
        TEACHER_REGULARIZATION="ema"
        ;;
    frozen)
        TEACHER_MODEL_SOURCE="legacy"
        TEACHER_REGULARIZATION="none"
        ;;
    fixed)
        if [[ -z "$TEACHER_MODEL_PATH" ]]; then
            echo "Error: TEACHER_MODE=fixed requires TEACHER_MODEL_PATH." >&2
            exit 1
        fi
        TEACHER_MODEL_SOURCE="fixed"
        TEACHER_REGULARIZATION="none"
        ;;
    coevolving)
        TEACHER_MODEL_SOURCE="actor"
        TEACHER_REGULARIZATION="none"
        ;;
    *)
        echo "Error: Unknown TEACHER_MODE=$TEACHER_MODE (expected: ema, frozen, fixed, coevolving)" >&2
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
OPD_TOKEN_MASK_PCT="${OPD_TOKEN_MASK_PCT:-0.0}"
OPD_TOKEN_MASK_METRIC="${OPD_TOKEN_MASK_METRIC:-loss}"
OPD_BUCKET_METRICS="${OPD_BUCKET_METRICS:-True}"
OPD_BUCKET_Q_LOW="${OPD_BUCKET_Q_LOW:-0.70}"
OPD_BUCKET_Q_HIGH="${OPD_BUCKET_Q_HIGH:-0.90}"
OPD_SELECTIVE_WEIGHT="${OPD_SELECTIVE_WEIGHT:-False}"
OPD_SELECTIVE_WEIGHT_MODE="${OPD_SELECTIVE_WEIGHT_MODE:-entropy_rkl_bucket}"
case "$OPD_SELECTIVE_WEIGHT_MODE" in
    entropy_rkl_bucket|risk_only_mask|random_mask|entropy_mask)
        ;;
    protect_risk_unprotect)
        echo "Error: OPD_SELECTIVE_WEIGHT_MODE=protect_risk_unprotect has been removed. Use risk_only_mask." >&2
        exit 1
        ;;
    *)
        echo "Error: Unknown OPD_SELECTIVE_WEIGHT_MODE=$OPD_SELECTIVE_WEIGHT_MODE (expected: entropy_rkl_bucket, risk_only_mask, random_mask, or entropy_mask)" >&2
        exit 1
        ;;
esac
OPD_SELECTIVE_WEIGHT_ENTROPY_LOW_Q="${OPD_SELECTIVE_WEIGHT_ENTROPY_LOW_Q:-0.40}"
OPD_SELECTIVE_WEIGHT_NLL_LOW_Q="${OPD_SELECTIVE_WEIGHT_NLL_LOW_Q:-0.40}"
OPD_SELECTIVE_WEIGHT_LOSS_LOW_Q="${OPD_SELECTIVE_WEIGHT_LOSS_LOW_Q:-0.40}"
OPD_SELECTIVE_WEIGHT_LOSS_MID_Q="${OPD_SELECTIVE_WEIGHT_LOSS_MID_Q:-0.50}"
OPD_RISK_MASK_TOP_P="${OPD_RISK_MASK_TOP_P:-0.30}"
OPD_SELECTIVE_WEIGHT_RISK_WAS_SET="${OPD_SELECTIVE_WEIGHT_RISK+x}"
OPD_SELECTIVE_WEIGHT_OTHER_WAS_SET="${OPD_SELECTIVE_WEIGHT_OTHER+x}"
OPD_SELECTIVE_WEIGHT_RISK="${OPD_SELECTIVE_WEIGHT_RISK:-1.00}"
OPD_SELECTIVE_WEIGHT_OTHER="${OPD_SELECTIVE_WEIGHT_OTHER:-1.00}"
if [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "risk_only_mask" || "$OPD_SELECTIVE_WEIGHT_MODE" == "random_mask" || "$OPD_SELECTIVE_WEIGHT_MODE" == "entropy_mask" ]]; then
    if [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "risk_only_mask" ]]; then
        OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE="${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE:-nll}"
    else
        OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE="${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE:-entropy}"
    fi
    OPD_SELECTIVE_WEIGHT_TIERED_PROTECT="${OPD_SELECTIVE_WEIGHT_TIERED_PROTECT:-False}"
    OPD_SELECTIVE_WEIGHT_NORMALIZE="${OPD_SELECTIVE_WEIGHT_NORMALIZE:-False}"
    OPD_SELECTIVE_WEIGHT_PROTECT_STRONG_Q="${OPD_SELECTIVE_WEIGHT_PROTECT_STRONG_Q:-0.25}"
    OPD_SELECTIVE_WEIGHT_ENTROPY_HIGH_Q="${OPD_SELECTIVE_WEIGHT_ENTROPY_HIGH_Q:-0.75}"
    OPD_SELECTIVE_WEIGHT_NLL_HIGH_Q="${OPD_SELECTIVE_WEIGHT_NLL_HIGH_Q:-0.80}"
    OPD_SELECTIVE_WEIGHT_LOSS_HIGH_Q="${OPD_SELECTIVE_WEIGHT_LOSS_HIGH_Q:-0.75}"
    OPD_SELECTIVE_WEIGHT_PROTECT_STRONG="${OPD_SELECTIVE_WEIGHT_PROTECT_STRONG:-0.25}"
    OPD_SELECTIVE_WEIGHT_PROTECT_MID="${OPD_SELECTIVE_WEIGHT_PROTECT_MID:-0.50}"
    OPD_SELECTIVE_WEIGHT_PROTECT_WEAK="${OPD_SELECTIVE_WEIGHT_PROTECT_WEAK:-0.75}"
    OPD_SELECTIVE_WEIGHT_PROTECT="${OPD_SELECTIVE_WEIGHT_PROTECT:-0.00}"
    OPD_SELECTIVE_WEIGHT_UNCLEAR="${OPD_SELECTIVE_WEIGHT_UNCLEAR:-0.00}"
    [[ -z "$OPD_SELECTIVE_WEIGHT_RISK_WAS_SET" ]] && OPD_SELECTIVE_WEIGHT_RISK="1.00"
    [[ -z "$OPD_SELECTIVE_WEIGHT_OTHER_WAS_SET" ]] && OPD_SELECTIVE_WEIGHT_OTHER="0.00"
else
    OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE="${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE:-entropy}"
    OPD_SELECTIVE_WEIGHT_TIERED_PROTECT="${OPD_SELECTIVE_WEIGHT_TIERED_PROTECT:-False}"
    OPD_SELECTIVE_WEIGHT_NORMALIZE="${OPD_SELECTIVE_WEIGHT_NORMALIZE:-True}"
    OPD_SELECTIVE_WEIGHT_PROTECT_STRONG_Q="${OPD_SELECTIVE_WEIGHT_PROTECT_STRONG_Q:-0.25}"
    OPD_SELECTIVE_WEIGHT_ENTROPY_HIGH_Q="${OPD_SELECTIVE_WEIGHT_ENTROPY_HIGH_Q:-0.75}"
    OPD_SELECTIVE_WEIGHT_NLL_HIGH_Q="${OPD_SELECTIVE_WEIGHT_NLL_HIGH_Q:-0.80}"
    OPD_SELECTIVE_WEIGHT_LOSS_HIGH_Q="${OPD_SELECTIVE_WEIGHT_LOSS_HIGH_Q:-0.75}"
    OPD_SELECTIVE_WEIGHT_PROTECT_STRONG="${OPD_SELECTIVE_WEIGHT_PROTECT_STRONG:-0.25}"
    OPD_SELECTIVE_WEIGHT_PROTECT_MID="${OPD_SELECTIVE_WEIGHT_PROTECT_MID:-0.50}"
    OPD_SELECTIVE_WEIGHT_PROTECT_WEAK="${OPD_SELECTIVE_WEIGHT_PROTECT_WEAK:-0.75}"
    OPD_SELECTIVE_WEIGHT_PROTECT="${OPD_SELECTIVE_WEIGHT_PROTECT:-0.50}"
    OPD_SELECTIVE_WEIGHT_UNCLEAR="${OPD_SELECTIVE_WEIGHT_UNCLEAR:-0.50}"
fi
OPD_TRAIN_METRICS="${OPD_TRAIN_METRICS:-True}"
OPD_TRAIN_METRICS_VERBOSE="${OPD_TRAIN_METRICS_VERBOSE:-False}"
OPD_SELECTIVE_METRICS_VERBOSE="${OPD_SELECTIVE_METRICS_VERBOSE:-False}"
OPD_METRICS_ENTROPY="${OPD_METRICS_ENTROPY:-False}"
OPD_TRACE_TOKEN="${OPD_TRACE_TOKEN:-False}"
OPD_TRACE_EVERY_N_STEPS="${OPD_TRACE_EVERY_N_STEPS:-5}"
OPD_TRACE_MAX_SAMPLES="${OPD_TRACE_MAX_SAMPLES:-4}"
OPD_TRACE_TOPK="${OPD_TRACE_TOPK:-$DISTILLATION_TOPK}"
OPD_TRACE_ENTROPY="${OPD_TRACE_ENTROPY:-True}"
OPD_TRACE_ONLY_FIRST_PPO_EPOCH="${OPD_TRACE_ONLY_FIRST_PPO_EPOCH:-True}"

# --- Training hyperparams ---
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$DEFAULT_PPO_MINI_BATCH_SIZE}"
ROLLOUT_N="${ROLLOUT_N:-4}"           # 4 = multi-rollout KD (no GRPO loss)
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"  # per rollout engine; training still uses trainer.n_gpus_per_node GPUs
LR="${LR:-1e-6}"
DONT_REPROMPT_ON_SELF_SUCCESS=True
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8192}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-5120}"
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-$DEFAULT_ROLLOUT_GPU_MEMORY_UTILIZATION}"
ACTOR_USE_DYNAMIC_BSZ=True
PPO_MAX_TOKEN_LEN_PER_GPU_WAS_SET="${PPO_MAX_TOKEN_LEN_PER_GPU+x}"
if [[ -z "$PPO_MAX_TOKEN_LEN_PER_GPU_WAS_SET" ]]; then
    if [[ -n "$DEFAULT_PPO_TOKEN_FACTOR" ]]; then
        PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_MODEL_LEN * DEFAULT_PPO_TOKEN_FACTOR))
    elif [[ "$MODEL_NAME_LC" == *"thinking"* ]]; then
        PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_MODEL_LEN * 2))
    else
        PPO_MAX_TOKEN_LEN_PER_GPU=$((MAX_MODEL_LEN * 4))
    fi
fi
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_LOGPROB_MICRO_BSZ}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_LOGPROB_MICRO_BSZ}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_MODEL_LEN * DEFAULT_ROLLOUT_BATCHED_FACTOR))}"
MAX_REPROMPT_LEN="${MAX_REPROMPT_LEN:-$MAX_MODEL_LEN}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-$DEFAULT_ACTOR_PARAM_OFFLOAD}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-$DEFAULT_ACTOR_OPTIMIZER_OFFLOAD}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-$DEFAULT_REF_PARAM_OFFLOAD}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-8}"
TRAINER_NNODES="${WORLD_SIZE:-1}"
TRAINER_SAVE_FREQ="${SAVE_FREQ:-50}"
TRAINER_TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TRAINER_MAX_ACTOR_CKPT_TO_KEEP="${TRAINER_MAX_ACTOR_CKPT_TO_KEEP:-$DEFAULT_MAX_ACTOR_CKPT_TO_KEEP}"
TRAINER_LOGGER="${TRAINER_LOGGER:-[\"console\",\"swanlab\"]}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"  # auto / disable / resume_path
FORCE_FRESH_START="${FORCE_FRESH_START:-False}"
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
POST_TRAIN_SYNC_TO_OSS="${POST_TRAIN_SYNC_TO_OSS:-False}"
POST_TRAIN_CLEAN_LOCAL="${POST_TRAIN_CLEAN_LOCAL:-False}"
POST_TRAIN_SYNC_ON_FAILURE="${POST_TRAIN_SYNC_ON_FAILURE:-False}"
POST_TRAIN_UPLOAD_WAIT_SECONDS="${POST_TRAIN_UPLOAD_WAIT_SECONDS:-1800}"
POST_TRAIN_UPLOAD_SWANLOG="${POST_TRAIN_UPLOAD_SWANLOG:-False}"
POST_TRAIN_CLEAN_LOGS="${POST_TRAIN_CLEAN_LOGS:-False}"
AUTO_TEE_LOG="${AUTO_TEE_LOG:-False}"
ROLLOUT_AGENT_NUM_WORKERS="${ROLLOUT_AGENT_NUM_WORKERS:-$TRAINER_N_GPUS_PER_NODE}"
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-0}"
ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-$DEFAULT_ACTOR_CKPT_SAVE_CONTENTS}"
ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-$ACTOR_CKPT_SAVE_CONTENTS}"
TRAINER_TOTAL_TRAINING_STEPS="${TRAINER_TOTAL_TRAINING_STEPS:-}"
CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE="${CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE:-True}"
CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD="${CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD:-False}"
CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE="${CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE:-True}"
CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE="${CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE:-False}"

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
if [[ "$DATASET_VERSION" == "quick" ]]; then
    TASK_TRAIN_FILE="${DATA_DIR}/train_1.5k.parquet"
    TASK_VAL_FILE="${DATA_DIR}/val.parquet"
elif [[ "$DATASET_VERSION" == "full" ]]; then
    TASK_TRAIN_FILE="${DATA_DIR}/train_5k.parquet"
    TASK_VAL_FILE="${DATA_DIR}/val.parquet"
else
    TASK_TRAIN_FILE="${DATA_DIR}/train.parquet"
    TASK_VAL_FILE="${DATA_DIR}/val.parquet"
fi
CUSTOM_DATASET_PATH="${RES_OPD_ROOT}/res_opd_dataset.py"

round_up_to_10() {
    local value="$1"
    if [[ "$value" -le 0 ]]; then
        echo 10
    else
        echo $(( ((value + 9) / 10) * 10 ))
    fi
}

auto_freq_from_steps() {
    local steps="$1"
    local parts="$2"
    local min_freq="$3"
    local raw
    local rounded
    if ! [[ "$steps" =~ ^[0-9]+$ ]] || [[ "$steps" -le 0 ]]; then
        echo "$min_freq"
        return 0
    fi
    raw=$(( (steps + parts - 1) / parts ))
    rounded="$(round_up_to_10 "$raw")"
    if [[ "$rounded" -lt "$min_freq" ]]; then
        rounded="$min_freq"
    fi
    if [[ "$rounded" -gt "$steps" ]]; then
        rounded="$steps"
    fi
    echo "$rounded"
}

TRAIN_SAMPLE_COUNT="$("$PYTHON_BIN" - "$TASK_TRAIN_FILE" <<'PY' 2>/dev/null || echo "?"
import sys
try:
    import pandas as pd
    print(len(pd.read_parquet(sys.argv[1])))
except Exception:
    print("?")
PY
)"
EXPECTED_STEPS_PER_EPOCH="?"
if [[ "$TRAIN_SAMPLE_COUNT" =~ ^[0-9]+$ && "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ && "$TRAIN_BATCH_SIZE" -gt 0 ]]; then
    # verl's PPO train dataloader uses drop_last=True.
    EXPECTED_STEPS_PER_EPOCH=$(( TRAIN_SAMPLE_COUNT / TRAIN_BATCH_SIZE ))
fi
EXPECTED_TOTAL_STEPS="?"
if [[ "$EXPECTED_STEPS_PER_EPOCH" =~ ^[0-9]+$ && "$TRAINER_TOTAL_EPOCHS" =~ ^[0-9]+$ ]]; then
    EXPECTED_TOTAL_STEPS=$(( EXPECTED_STEPS_PER_EPOCH * TRAINER_TOTAL_EPOCHS ))
fi
if [[ -n "$TRAINER_TOTAL_TRAINING_STEPS" ]]; then
    EXPECTED_TOTAL_STEPS="$TRAINER_TOTAL_TRAINING_STEPS"
fi
EFFECTIVE_ROLLOUT_BATCH_SIZE="?"
EFFECTIVE_PPO_BATCH_SIZE="?"
if [[ "$TRAIN_BATCH_SIZE" =~ ^[0-9]+$ && "$ROLLOUT_N" =~ ^[0-9]+$ ]]; then
    EFFECTIVE_ROLLOUT_BATCH_SIZE=$(( TRAIN_BATCH_SIZE * ROLLOUT_N ))
fi
if [[ "$PPO_MINI_BATCH_SIZE" =~ ^[0-9]+$ && "$ROLLOUT_N" =~ ^[0-9]+$ ]]; then
    EFFECTIVE_PPO_BATCH_SIZE=$(( PPO_MINI_BATCH_SIZE * ROLLOUT_N ))
fi
if [[ "$TRAINER_SAVE_FREQ" == "auto" ]]; then
    TRAINER_SAVE_FREQ="$(auto_freq_from_steps "$EXPECTED_STEPS_PER_EPOCH" 4 20)"
fi

# --- Experiment naming ---
EPOCH_TAG="e${TRAINER_TOTAL_EPOCHS}"
case "$ALPHA" in
    1|1.0|1.00)
        LOSS_TAG="rkl"
        ;;
    0.5|0.50|.5)
        LOSS_TAG="jsd"
        ;;
    0|0.0|0.00)
        LOSS_TAG="fkl"
        ;;
    *)
        LOSS_TAG="a${ALPHA}"
        ;;
esac
case "$DATASET_VERSION" in
    full)
        DATASET_TAG="full5k"
        ;;
    quick)
        DATASET_TAG="quick1p5k"
        ;;
    *)
        DATASET_TAG="legacy"
        ;;
esac
format_prob_tag() {
    "$PYTHON_BIN" - "$1" <<'PY' 2>/dev/null || echo "${1//./p}"
import sys
print(str(int(round(float(sys.argv[1]) * 100))))
PY
}
NAME_TAGS=("${TEACHER_MODE}" "${LOSS_TAG}")
if [[ "$TEACHER_MODE" == "fixed" ]]; then
    TEACHER_MODEL_NAME="$(basename "$TEACHER_MODEL_PATH")"
    NAME_TAGS+=("teacher${TEACHER_MODEL_NAME}")
fi
case "${OPD_SELECTIVE_WEIGHT:-False}" in
    True|true|TRUE|1|yes|YES|y|Y)
        case "$OPD_SELECTIVE_WEIGHT_MODE" in
            risk_only_mask)
                NAME_TAGS+=("riskmask" "${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE}" "p$(format_prob_tag "$OPD_RISK_MASK_TOP_P")")
                ;;
            random_mask)
                NAME_TAGS+=("randommask" "p$(format_prob_tag "$OPD_RISK_MASK_TOP_P")")
                ;;
            entropy_mask)
                NAME_TAGS+=("entropymask" "p$(format_prob_tag "$OPD_RISK_MASK_TOP_P")")
                ;;
            *)
                NAME_TAGS+=("sw" "${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE}")
                ;;
        esac
        ;;
esac
if [[ "$OPD_TOKEN_MASK_PCT" != "0" && "$OPD_TOKEN_MASK_PCT" != "0.0" && "$OPD_TOKEN_MASK_PCT" != "0.00" ]]; then
    NAME_TAGS+=("maskp$(format_prob_tag "$OPD_TOKEN_MASK_PCT")")
fi
NAME_TAGS+=("b${TRAIN_BATCH_SIZE}" "rn${ROLLOUT_N}" "$DATASET_TAG" "$EPOCH_TAG")
IFS=-
NAME_SUFFIX="${NAME_TAGS[*]}"
unset IFS
if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
    : # Use externally provided EXPERIMENT_NAME
else
    EXPERIMENT_NAME="Res-OPD-${MODEL_NAME}-orig-sr${STUDENT_RATIO}-tr${TEACHER_RATIO}-a${ALPHA}-${NAME_SUFFIX}"
fi
PROJECT_NAME="Res-OPD"
TRAINER_DEFAULT_LOCAL_DIR="${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAINER_ROLLOUT_DATA_DIR="${RES_OPD_ROOT}/rollouts/${EXPERIMENT_NAME}"
OPD_TRACE_DIR="${OPD_TRACE_DIR:-${RES_OPD_ROOT}/traces/${EXPERIMENT_NAME}}"
OPD_MINI_EVAL_TRACE="${OPD_MINI_EVAL_TRACE:-False}"
OPD_MINI_EVAL_GENERATION_DIR="${OPD_MINI_EVAL_GENERATION_DIR:-${RES_OPD_ROOT}/mini_eval_generations/${EXPERIMENT_NAME}}"
OPD_MINI_EVAL_MAX_SAMPLES="${OPD_MINI_EVAL_MAX_SAMPLES:-100}"
OPD_MINI_EVAL_TEST_FREQ="${OPD_MINI_EVAL_TEST_FREQ:-500}"
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-False}"
VALIDATION_METRIC_MODE="${VALIDATION_METRIC_MODE:-mean_only}"
export EXPERIMENT="$EXPERIMENT_NAME"
WATCHER_SCRIPT="${RES_OPD_ROOT}/scripts/ckpt_upload_watcher.sh"
WATCHER_PID_FILE="${TRAINER_DEFAULT_LOCAL_DIR}/.watcher.pid"

if [[ "$AUTO_TEE_LOG" =~ ^(True|true|TRUE|1|yes|YES|y|Y)$ && -z "${RES_OPD_TEE_ACTIVE:-}" ]]; then
    mkdir -p "${RES_OPD_ROOT}/logs"
    export RES_OPD_TEE_ACTIVE=1
    exec > >(tee -a "${RES_OPD_ROOT}/logs/${EXPERIMENT_NAME}.log") 2>&1
fi

TEACHER_MODEL_PATH_OVERRIDE=()
if [[ "$TEACHER_MODE" == "fixed" ]]; then
    TEACHER_MODEL_PATH_OVERRIDE=(
        actor_rollout_ref.actor.self_distillation.teacher_model_path="$TEACHER_MODEL_PATH"
    )
fi

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

csv_to_hydra_list() {
    local csv="$1"
    local items item out first
    IFS=',' read -ra items <<< "$csv"
    out="["
    first=true
    for item in "${items[@]}"; do
        item="${item//[[:space:]]/}"
        [[ -z "$item" ]] && continue
        case "$item" in
            model|optimizer|extra|hf_model)
                ;;
            *)
                echo "Error: unsupported checkpoint content '${item}' in '${csv}'." >&2
                echo "Supported: model, optimizer, extra, hf_model" >&2
                exit 1
                ;;
        esac
        if $first; then
            first=false
        else
            out+=","
        fi
        out+="'${item}'"
    done
    out+="]"
    if [[ "$out" == "[]" ]]; then
        echo "Error: checkpoint content list is empty: '${csv}'." >&2
        exit 1
    fi
    echo "$out"
}

ACTOR_CKPT_SAVE_CONTENTS_HYDRA="$(csv_to_hydra_list "$ACTOR_CKPT_SAVE_CONTENTS")"
ACTOR_CKPT_LOAD_CONTENTS_HYDRA="$(csv_to_hydra_list "$ACTOR_CKPT_LOAD_CONTENTS")"

if is_truthy "$OPD_MINI_EVAL_TRACE"; then
    TRAINER_VALIDATION_DATA_DIR="$OPD_MINI_EVAL_GENERATION_DIR"
    if [[ -n "${TEST_FREQ:-}" ]]; then
        TRAINER_TEST_FREQ="$TEST_FREQ"
    elif [[ "$OPD_MINI_EVAL_TEST_FREQ" == "auto" ]]; then
        TRAINER_TEST_FREQ="$(auto_freq_from_steps "$EXPECTED_STEPS_PER_EPOCH" 4 20)"
    else
        TRAINER_TEST_FREQ="$OPD_MINI_EVAL_TEST_FREQ"
    fi
    DATA_VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-$OPD_MINI_EVAL_MAX_SAMPLES}"
else
    TRAINER_VALIDATION_DATA_DIR="${TRAINER_VALIDATION_DATA_DIR:-null}"
    TRAINER_TEST_FREQ="${TEST_FREQ:-500}"
    DATA_VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
fi
if is_truthy "$OPD_SELECTIVE_WEIGHT"; then
    OPD_METRICS_ENTROPY=True
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
    if [[ "$ckpt_dir_name" == Res-OPD-Qwen3VL-2B-Instruct-* ]]; then
        # Backward-compatible path for existing Instruct checkpoints.
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

count_actor_model_shards_for_upload() {
    local step_dir="$1"
    local actor_dir="${step_dir}/actor"
    if [[ ! -d "$actor_dir" ]]; then
        echo 0
        return 0
    fi
    find "$actor_dir" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' | wc -l | tr -d '[:space:]'
}

expected_actor_world_size_for_upload() {
    local step_dir="$1"
    local fsdp_config="${step_dir}/actor/fsdp_config.json"
    if [[ ! -f "$fsdp_config" ]]; then
        echo ""
        return 0
    fi
    grep -o '"world_size"[[:space:]]*:[[:space:]]*[0-9]\+' "$fsdp_config" 2>/dev/null \
        | grep -o '[0-9]\+' \
        | head -n 1 || true
}

checkpoint_upload_missing_reason() {
    local step_dir="$1"
    local actor_dir="${step_dir}/actor"
    local fsdp_config="${actor_dir}/fsdp_config.json"
    local hf_config="${actor_dir}/huggingface/config.json"
    local shard_count expected_world_size

    if [[ ! -f "${step_dir}/data.pt" ]]; then
        echo "missing data.pt"
        return 0
    fi
    if [[ ! -d "$actor_dir" ]]; then
        echo "missing actor/"
        return 0
    fi
    if [[ ! -f "$fsdp_config" ]]; then
        echo "missing actor/fsdp_config.json"
        return 0
    fi
    if [[ ! -f "$hf_config" ]]; then
        echo "missing actor/huggingface/config.json"
        return 0
    fi

    shard_count="$(count_actor_model_shards_for_upload "$step_dir")"
    if [[ "$shard_count" == "0" ]]; then
        echo "missing actor model shards"
        return 0
    fi

    expected_world_size="$(expected_actor_world_size_for_upload "$step_dir")"
    if [[ -n "$expected_world_size" && "$shard_count" -lt "$expected_world_size" ]]; then
        echo "only ${shard_count}/${expected_world_size} actor model shards present"
        return 0
    fi

    return 1
}

all_checkpoint_steps_uploaded() {
    local step_dir found_uploaded=false missing_reason=""
    local found=false
    for step_dir in "${TRAINER_DEFAULT_LOCAL_DIR}"/global_step_*; do
        [[ -d "$step_dir" ]] || continue
        found=true
        if [[ -f "${step_dir}/.oss_uploaded" ]]; then
            found_uploaded=true
            continue
        fi
        if ! missing_reason="$(checkpoint_upload_missing_reason "$step_dir")"; then
            return 1
        fi
        if [[ -d "${step_dir}/actor" || -f "${step_dir}/data.pt" ]]; then
            echo "Skipping non-uploadable checkpoint step during post-train wait: $(basename "$step_dir") (${missing_reason})"
        fi
    done
    $found && $found_uploaded
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
            if ! wait_for_checkpoint_uploads 300; then
                echo "ERROR: checkpoint upload did not complete after one-shot scan." >&2
                echo "Checkpoint dir: $TRAINER_DEFAULT_LOCAL_DIR" >&2
                echo "Watcher log: ${RES_OPD_ROOT}/logs/ckpt_watcher_${EXPERIMENT_NAME}.log" >&2
                return 1
            fi
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
        env \
            OSS_BASE="$OSS_BASE" \
            CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE="$CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE" \
            CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD="$CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD" \
            CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE="$CKPT_WATCHER_DELETE_MERGED_ON_UPLOAD_FAILURE" \
            CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE="$CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE" \
            nohup bash "$WATCHER_SCRIPT" --watch-dir "$TRAINER_DEFAULT_LOCAL_DIR" > /dev/null 2>&1 &
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
echo "Model size prof.: $MODEL_SIZE_PROFILE"
echo "Degradation mode: $DEGRADATION_MODE"
echo "Legacy px args:   student=$STUDENT_PX teacher=$TEACHER_PX target=$TARGET_PX (metadata only)"
echo "Student ratio:    $STUDENT_RATIO (original mode)"
echo "Teacher ratio:    $TEACHER_RATIO (original mode)"
echo "Teacher mode:     $TEACHER_MODE (src=$TEACHER_MODEL_SOURCE, reg=$TEACHER_REGULARIZATION, rate=$TEACHER_UPDATE_RATE)"
if [[ "$TEACHER_MODE" == "fixed" ]]; then
    echo "Teacher model:    $TEACHER_MODEL_PATH"
fi
echo "Alpha (loss):     $ALPHA (0.5=JSD, 1.0=RKL, 0.0=FKL)"
echo "Selective veto:   $OPD_SELECTIVE_VETO (top_p=$OPD_SELECTIVE_VETO_TOP_P, min_score=$OPD_SELECTIVE_VETO_MIN_SCORE, normalize=$OPD_SELECTIVE_VETO_NORMALIZE)"
echo "Token mask:       pct=$OPD_TOKEN_MASK_PCT metric=$OPD_TOKEN_MASK_METRIC (bucket_metrics=$OPD_BUCKET_METRICS, bucket_q=${OPD_BUCKET_Q_LOW}/${OPD_BUCKET_Q_HIGH})"
SELECTIVE_WEIGHT_RISK_LABEL="risk"
if [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "risk_only_mask" || "$OPD_SELECTIVE_WEIGHT_MODE" == "random_mask" || "$OPD_SELECTIVE_WEIGHT_MODE" == "entropy_mask" ]]; then
    SELECTIVE_WEIGHT_RISK_LABEL="selected"
fi
if [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "risk_only_mask" || "$OPD_SELECTIVE_WEIGHT_MODE" == "random_mask" || "$OPD_SELECTIVE_WEIGHT_MODE" == "entropy_mask" ]]; then
    MASK_SCORE_DESC="$OPD_SELECTIVE_WEIGHT_MODE"
    [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "risk_only_mask" ]] && MASK_SCORE_DESC="rank(RKL)*rank(${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE})"
    [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "entropy_mask" ]] && MASK_SCORE_DESC="rank(student_entropy)"
    [[ "$OPD_SELECTIVE_WEIGHT_MODE" == "random_mask" ]] && MASK_SCORE_DESC="uniform_random"
    echo "Selective weight: $OPD_SELECTIVE_WEIGHT (mode=$OPD_SELECTIVE_WEIGHT_MODE, mask_score=$MASK_SCORE_DESC, top_p=$OPD_RISK_MASK_TOP_P, normalize=$OPD_SELECTIVE_WEIGHT_NORMALIZE, weights ${SELECTIVE_WEIGHT_RISK_LABEL}/other=${OPD_SELECTIVE_WEIGHT_RISK}/${OPD_SELECTIVE_WEIGHT_OTHER})"
else
    echo "Selective weight: $OPD_SELECTIVE_WEIGHT (mode=$OPD_SELECTIVE_WEIGHT_MODE, uncertainty=$OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE, tiered=$OPD_SELECTIVE_WEIGHT_TIERED_PROTECT, normalize=$OPD_SELECTIVE_WEIGHT_NORMALIZE, entropy_q=${OPD_SELECTIVE_WEIGHT_ENTROPY_LOW_Q}/${OPD_SELECTIVE_WEIGHT_ENTROPY_HIGH_Q}, nll_q=${OPD_SELECTIVE_WEIGHT_NLL_LOW_Q}/${OPD_SELECTIVE_WEIGHT_NLL_HIGH_Q}, loss_q=${OPD_SELECTIVE_WEIGHT_LOSS_LOW_Q}/${OPD_SELECTIVE_WEIGHT_LOSS_MID_Q}/${OPD_SELECTIVE_WEIGHT_LOSS_HIGH_Q}, weights protect/unclear/${SELECTIVE_WEIGHT_RISK_LABEL}/other=${OPD_SELECTIVE_WEIGHT_PROTECT}/${OPD_SELECTIVE_WEIGHT_UNCLEAR}/${OPD_SELECTIVE_WEIGHT_RISK}/${OPD_SELECTIVE_WEIGHT_OTHER})"
fi
echo "Rollout N:        $ROLLOUT_N (1=pure KD, >1=GRPO+KD)"
echo "Learning rate:    $LR"
echo "Batch size:       $TRAIN_BATCH_SIZE"
echo "PPO mini batch:   $PPO_MINI_BATCH_SIZE"
echo "Effective batches: rollout=$EFFECTIVE_ROLLOUT_BATCH_SIZE ppo=$EFFECTIVE_PPO_BATCH_SIZE (batch * rollout_n)"
echo "Max lengths:      prompt=$MAX_PROMPT_LENGTH response=$MAX_RESPONSE_LENGTH model=$MAX_MODEL_LEN reprompt=$MAX_REPROMPT_LEN"
echo "Trainer GPUs:     $TRAINER_N_GPUS_PER_NODE/node x $TRAINER_NNODES node(s)"
echo "Rollout workers:  $ROLLOUT_AGENT_NUM_WORKERS"
echo "Rollout TP/engine:$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE (TP=1 here does not mean single-GPU training)"
echo "Rollout max batched tokens: $ROLLOUT_MAX_NUM_BATCHED_TOKENS"
echo "PPO max tokens/GPU: $PPO_MAX_TOKEN_LEN_PER_GPU"
echo "Logprob micro bsz: rollout=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU ref=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU"
echo "OPD metrics:      $OPD_TRAIN_METRICS (entropy curve=$OPD_METRICS_ENTROPY, train_verbose=$OPD_TRAIN_METRICS_VERBOSE, selective_verbose=$OPD_SELECTIVE_METRICS_VERBOSE)"
echo "OPD token trace:  $OPD_TRACE_TOKEN (every ${OPD_TRACE_EVERY_N_STEPS} steps, max ${OPD_TRACE_MAX_SAMPLES}/rank, topk=${OPD_TRACE_TOPK})"
echo "Trace dir:        $OPD_TRACE_DIR"
echo "Mini-eval trace:  $OPD_MINI_EVAL_TRACE (test_freq=$TRAINER_TEST_FREQ, max_samples=$DATA_VAL_MAX_SAMPLES, val_n=$VAL_N, do_sample=$VAL_DO_SAMPLE)"
echo "Mini-eval gen:    $TRAINER_VALIDATION_DATA_DIR"
echo "Val metric mode:  $VALIDATION_METRIC_MODE"
echo "Resume mode:      $TRAINER_RESUME_MODE (force fresh=$FORCE_FRESH_START)"
echo "OSS base:         $OSS_BASE"
echo "Post-train OSS:   sync=$POST_TRAIN_SYNC_TO_OSS clean_local=$POST_TRAIN_CLEAN_LOCAL"
echo "Ckpt contents:    save=$ACTOR_CKPT_SAVE_CONTENTS load=$ACTOR_CKPT_LOAD_CONTENTS"
echo "Ckpt watcher:     prune_optim=$CKPT_WATCHER_PRUNE_OPTIMIZER_BEFORE_MERGE keep_fsdp=$CKPT_WATCHER_KEEP_LOCAL_FSDP_AFTER_UPLOAD delete_fsdp_on_merge_fail=$CKPT_WATCHER_DELETE_FSDP_ON_MERGE_FAILURE"
echo "Experiment:       $EXPERIMENT_NAME"
echo "Data:             $TASK_TRAIN_FILE"
echo "Train samples:    $TRAIN_SAMPLE_COUNT"
echo "Expected steps/epoch: $EXPECTED_STEPS_PER_EPOCH (drop_last=True)"
echo "Expected total steps: $EXPECTED_TOTAL_STEPS"
if [[ -n "$TRAINER_TOTAL_TRAINING_STEPS" ]]; then
    echo "Step override:    trainer.total_training_steps=$TRAINER_TOTAL_TRAINING_STEPS"
fi
echo "Dataset class:    ResOPDDataset ($CUSTOM_DATASET_PATH)"
echo "Checkpoints:      $TRAINER_DEFAULT_LOCAL_DIR"
echo "============================================================"

# =============================================================================
# LAUNCH TRAINING
# =============================================================================
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
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_REPROMPT_LEN \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    actor_rollout_ref.actor.self_distillation.teacher_model_source=$TEACHER_MODEL_SOURCE \
    ${TEACHER_MODEL_PATH_OVERRIDE[@]+"${TEACHER_MODEL_PATH_OVERRIDE[@]}"} \
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
    actor_rollout_ref.actor.self_distillation.token_mask_pct=$OPD_TOKEN_MASK_PCT \
    actor_rollout_ref.actor.self_distillation.token_mask_metric=$OPD_TOKEN_MASK_METRIC \
    actor_rollout_ref.actor.self_distillation.selective_bucket_metrics_enabled=$OPD_BUCKET_METRICS \
    actor_rollout_ref.actor.self_distillation.selective_bucket_q_low=$OPD_BUCKET_Q_LOW \
    actor_rollout_ref.actor.self_distillation.selective_bucket_q_high=$OPD_BUCKET_Q_HIGH \
    actor_rollout_ref.actor.self_distillation.selective_weight_enabled=$OPD_SELECTIVE_WEIGHT \
    actor_rollout_ref.actor.self_distillation.selective_weight_mode=$OPD_SELECTIVE_WEIGHT_MODE \
    actor_rollout_ref.actor.self_distillation.selective_weight_uncertainty_mode=$OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE \
    actor_rollout_ref.actor.self_distillation.selective_weight_tiered_protect=$OPD_SELECTIVE_WEIGHT_TIERED_PROTECT \
    actor_rollout_ref.actor.self_distillation.selective_weight_normalize=$OPD_SELECTIVE_WEIGHT_NORMALIZE \
    actor_rollout_ref.actor.self_distillation.selective_weight_risk_top_p=$OPD_RISK_MASK_TOP_P \
    actor_rollout_ref.actor.self_distillation.selective_weight_protect_strong_q=$OPD_SELECTIVE_WEIGHT_PROTECT_STRONG_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_entropy_low_q=$OPD_SELECTIVE_WEIGHT_ENTROPY_LOW_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_entropy_high_q=$OPD_SELECTIVE_WEIGHT_ENTROPY_HIGH_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_nll_low_q=$OPD_SELECTIVE_WEIGHT_NLL_LOW_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_nll_high_q=$OPD_SELECTIVE_WEIGHT_NLL_HIGH_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_loss_low_q=$OPD_SELECTIVE_WEIGHT_LOSS_LOW_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_loss_mid_q=$OPD_SELECTIVE_WEIGHT_LOSS_MID_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_loss_high_q=$OPD_SELECTIVE_WEIGHT_LOSS_HIGH_Q \
    actor_rollout_ref.actor.self_distillation.selective_weight_protect_strong=$OPD_SELECTIVE_WEIGHT_PROTECT_STRONG \
    actor_rollout_ref.actor.self_distillation.selective_weight_protect_mid=$OPD_SELECTIVE_WEIGHT_PROTECT_MID \
    actor_rollout_ref.actor.self_distillation.selective_weight_protect_weak=$OPD_SELECTIVE_WEIGHT_PROTECT_WEAK \
    actor_rollout_ref.actor.self_distillation.selective_weight_protect=$OPD_SELECTIVE_WEIGHT_PROTECT \
    actor_rollout_ref.actor.self_distillation.selective_weight_unclear=$OPD_SELECTIVE_WEIGHT_UNCLEAR \
    actor_rollout_ref.actor.self_distillation.selective_weight_risk=$OPD_SELECTIVE_WEIGHT_RISK \
    actor_rollout_ref.actor.self_distillation.selective_weight_other=$OPD_SELECTIVE_WEIGHT_OTHER \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.self_distillation.train_metrics_enabled=$OPD_TRAIN_METRICS \
    actor_rollout_ref.actor.self_distillation.train_metrics_verbose=$OPD_TRAIN_METRICS_VERBOSE \
    actor_rollout_ref.actor.self_distillation.selective_metrics_verbose=$OPD_SELECTIVE_METRICS_VERBOSE \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
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
    trainer.total_training_steps="${TRAINER_TOTAL_TRAINING_STEPS:-null}" \
    +trainer.save_at_epoch_end="${SAVE_AT_EPOCH_END:-True}" \
    +trainer.test_at_epoch_end="${TEST_AT_EPOCH_END:-False}" \
    actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
    +trainer.validation_metric_mode="$VALIDATION_METRIC_MODE" \
    actor_rollout_ref.actor.checkpoint.save_contents="$ACTOR_CKPT_SAVE_CONTENTS_HYDRA" \
    actor_rollout_ref.actor.checkpoint.load_contents="$ACTOR_CKPT_LOAD_CONTENTS_HYDRA" \
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

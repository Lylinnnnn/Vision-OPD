#!/bin/bash

set -eo pipefail

# =============================================================================
# Res-OPD Training Script
# Resolution-Aware On-Policy Self-Distillation
#
# All Vision-OPD features preserved. Additional knobs for resolution distillation:
#   - STUDENT_PX:          student resolution (0 = no degradation)
#   - TARGET_PX:           target resolution for upsampling
#   - TEACHER_MODE:        ema / frozen / coevolving
#   - ALPHA:               loss interpolation (0.5=JSD, 1.0=RKL, 0.0=FKL)
#   - ROLLOUT_N:           number of rollouts per sample (1=pure KD, 8=GRPO+KD)
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
TEACHER_PX="${TEACHER_PX:-0}"         # 0 = use target_px as-is (no teacher degradation)

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
ROLLOUT_AGENT_NUM_WORKERS=8
DATA_DATALOADER_NUM_WORKERS="${DATA_DATALOADER_NUM_WORKERS:-0}"

# --- Data paths ---
DATA_DIR="${RES_OPD_ROOT}/data"
TASK_TRAIN_FILE="${DATA_DIR}/train.parquet"
CUSTOM_DATASET_PATH="${RES_OPD_ROOT}/res_opd_dataset.py"

# --- Experiment naming ---
MODEL_NAME=$(basename "$MODEL_PATH")
EPOCH_TAG="e${TRAINER_TOTAL_EPOCHS}"
if [[ -n "${EXPERIMENT_NAME:-}" ]]; then
    : # Use externally provided EXPERIMENT_NAME
elif [[ "$TEACHER_PX" -gt 0 && "$TEACHER_PX" -lt "$TARGET_PX" ]]; then
    EXPERIMENT_NAME="Res-OPD-${MODEL_NAME}-s${STUDENT_PX}-t${TEACHER_PX}-a${ALPHA}-${TEACHER_MODE}-${EPOCH_TAG}"
else
    EXPERIMENT_NAME="Res-OPD-${MODEL_NAME}-s${STUDENT_PX}-a${ALPHA}-${TEACHER_MODE}-${EPOCH_TAG}"
fi
PROJECT_NAME="Res-OPD"
TRAINER_DEFAULT_LOCAL_DIR="${RES_OPD_ROOT}/checkpoints/${EXPERIMENT_NAME}"
TRAINER_ROLLOUT_DATA_DIR="${RES_OPD_ROOT}/rollouts/${EXPERIMENT_NAME}"
mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"

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
# AUTO-START CHECKPOINT UPLOAD WATCHER
# =============================================================================
WATCHER_SCRIPT="${RES_OPD_ROOT}/scripts/ckpt_upload_watcher.sh"
if [[ -f "$WATCHER_SCRIPT" ]]; then
    if ! pgrep -f "ckpt_upload_watcher.sh" > /dev/null 2>&1; then
        echo "Starting ckpt_upload_watcher in background ..."
        nohup bash "$WATCHER_SCRIPT" > /dev/null 2>&1 &
        echo "  Watcher PID: $!"
    else
        echo "ckpt_upload_watcher already running (PID: $(pgrep -f 'ckpt_upload_watcher.sh' | head -1))"
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
if [[ ! -f "$CUSTOM_DATASET_PATH" ]]; then
    echo "Error: Custom dataset not found at $CUSTOM_DATASET_PATH" >&2
    exit 1
fi

echo "============================================================"
echo " Res-OPD Training"
echo "============================================================"
echo "Model:            $MODEL_PATH"
echo "Student px:       $STUDENT_PX (0=no degradation)"
echo "Teacher px:       $TEACHER_PX (0=use target_px as-is)"
echo "Target px:        $TARGET_PX"
echo "Teacher mode:     $TEACHER_MODE (src=$TEACHER_MODEL_SOURCE, reg=$TEACHER_REGULARIZATION, rate=$TEACHER_UPDATE_RATE)"
echo "Alpha (loss):     $ALPHA (0.5=JSD, 1.0=RKL, 0.0=FKL)"
echo "Rollout N:        $ROLLOUT_N (1=pure KD, >1=GRPO+KD)"
echo "Learning rate:    $LR"
echo "Batch size:       $TRAIN_BATCH_SIZE"
echo "Experiment:       $EXPERIMENT_NAME"
echo "Data:             $TASK_TRAIN_FILE"
echo "Dataset class:    ResOPDDataset ($CUSTOM_DATASET_PATH)"
echo "Checkpoints:      $TRAINER_DEFAULT_LOCAL_DIR"
echo "============================================================"

# =============================================================================
# LAUNCH TRAINING
# =============================================================================
PYTHON_BIN="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3"
"$PYTHON_BIN" -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="[\"$TASK_TRAIN_FILE\"]" \
    data.val_files="[\"${RES_OPD_ROOT}/data/val.parquet\"]" \
    data.val_batch_size=50 \
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
    actor_rollout_ref.actor.calculate_entropy=False \
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
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
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
    trainer.test_freq="${TEST_FREQ:-20}" \
    trainer.save_at_epoch_end="${SAVE_AT_EPOCH_END:-True}" \
    trainer.test_at_epoch_end="${TEST_AT_EPOCH_END:-False}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-6}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${EXTRA_ARGS[@]}"

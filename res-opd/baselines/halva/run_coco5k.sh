#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <qwen2b|qwen4b|qwen8b|internvl2b> <prepare|preflight|train|export|eval|smoke>}"
PHASE="${2:?Specify phase}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
DATA_ROOT="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"
WORK_ROOT="${HALVA_WORK_ROOT:-$ROOT/checkpoints/halva_coco5k}"
RUN_NAME="${HALVA_RUN_NAME:-halva_coco5k}"
TRAIN_PY="${HALVA_TRAIN_PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/rlsf-v-lf/bin/python3}"
EVAL_PY="${HALVA_EVAL_PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"

case "$MODEL" in
    qwen2b) MODEL_PATH="${HALVA_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}" ;;
    qwen4b) MODEL_PATH="${HALVA_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-4B-Instruct}" ;;
    qwen8b) MODEL_PATH="${HALVA_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-8B-Instruct}" ;;
    internvl2b) MODEL_PATH="${HALVA_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/internvl/InternVL3-2B-hf}" ;;
    *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac

MODEL_DIR="$WORK_ROOT/$RUN_NAME/$MODEL"
QUESTIONS="${HALVA_QUESTIONS:-$ROOT/checkpoints/rlsf_v_coco5k/questions.jsonl}"
CAPTIONS="${HALVA_CAPTIONS:-$DATA_ROOT/COCO/annotations/captions_train2017.json}"
INSTANCES="${HALVA_INSTANCES:-$DATA_ROOT/COCO/annotations/instances_train2017.json}"
PAIRS="$MODEL_DIR/pairs.jsonl"
ADAPTER="$MODEL_DIR/adapter"
MERGED="$MODEL_DIR/merged"
OUT="$ROOT/eval_results/external_baselines/full/$(basename "$MODEL_PATH")/train5000_test1000_original_sr1p0/${RUN_NAME}_official_v2"

prepare() {
    test -s "$QUESTIONS" || { echo "Missing COCO 5k questions: $QUESTIONS" >&2; exit 1; }
    test -s "$CAPTIONS" || { echo "Missing COCO train captions: $CAPTIONS" >&2; exit 1; }
    test -s "$INSTANCES" || { echo "Missing COCO train instances: $INSTANCES" >&2; exit 1; }
    test ! -e "$PAIRS" || { echo "Pairs already exist: $PAIRS" >&2; exit 1; }
    "$TRAIN_PY" "$SCRIPT_DIR/prepare.py" --questions "$QUESTIONS" \
        --captions "$CAPTIONS" --instances "$INSTANCES" --output "$PAIRS" \
        --max-samples "${HALVA_MAX_SAMPLES:-0}"
}

preflight() {
    test -s "$PAIRS" || { echo "Run prepare first: $PAIRS" >&2; exit 1; }
    test -s "$MODEL_PATH/config.json" || { echo "Missing model config: $MODEL_PATH" >&2; exit 1; }
    "$TRAIN_PY" "$SCRIPT_DIR/train.py" --model "$MODEL" --model-path "$MODEL_PATH" \
        --pairs "$PAIRS" --output-dir "$ADAPTER" --preflight-only
}

train() {
    preflight
    "$TRAIN_PY" "$SCRIPT_DIR/train.py" --model "$MODEL" --model-path "$MODEL_PATH" \
        --pairs "$PAIRS" --output-dir "$ADAPTER" \
        --epochs "${HALVA_EPOCHS:-1}" --max-steps "${HALVA_MAX_STEPS:-0}" \
        --grad-accum "${HALVA_GRAD_ACCUM:-16}" --max-length "${HALVA_MAX_LENGTH:-4096}"
}

export_model() {
    "$TRAIN_PY" "$SCRIPT_DIR/export.py" --model "$MODEL" --model-path "$MODEL_PATH" \
        --adapter "$ADAPTER" --output-dir "$MERGED"
}

evaluate() {
    test -s "$MERGED/config.json" || { echo "Run export first: $MERGED" >&2; exit 1; }
    cd "$REPO"
    PYTHON_BIN="$EVAL_PY" RES_OPD_DATA_ROOT="$DATA_ROOT" EVAL_OUTPUT_DIR="$OUT" \
    DATASET_VERSION=full DEGRADATION_MODE=original STUDENT_RATIO=1.0 \
    CHAIR_PROMPT='Describe the image.' CHAIR_MAX_SAMPLES="${HALVA_EVAL_MAX_SAMPLES:-1000}" \
    POPE_MAX_SAMPLES="${HALVA_AUX_MAX_SAMPLES:-0}" AMBER_MAX_SAMPLES="${HALVA_AUX_MAX_SAMPLES:-0}" \
    AMBER_ROOT="$DATA_ROOT/AMBER" AMBER_SKIP_OFFICIAL_EVAL="${HALVA_SKIP_AMBER_METRICS:-False}" \
    AMBER_OFFICIAL_EVAL_WORKERS="${HALVA_AMBER_WORKERS:-16}" \
    EVAL_SHARD_COUNT="${HALVA_EVAL_SHARDS:-8}" GPU_LIST="${HALVA_GPU_LIST:-0,1,2,3,4,5,6,7}" \
    VLLM_BASE_PORT="${HALVA_BASE_PORT:-35400}" VLLM_PORT_CLEANUP=False \
    VLLM_STARTUP_TIMEOUT="${HALVA_VLLM_TIMEOUT:-3600}" \
    bash "$ROOT/scripts/eval_after_merge_sharded_8gpu.sh" "$MERGED" 0 instruct chair,pope,amber
}

case "$PHASE" in
    prepare) prepare ;;
    preflight) preflight ;;
    train) train ;;
    export) export_model ;;
    eval) evaluate ;;
    smoke)
        test "${HALVA_RUN_NAME:-}" != "" || { echo "Set a unique HALVA_RUN_NAME for smoke" >&2; exit 2; }
        HALVA_MAX_SAMPLES=2
        HALVA_MAX_STEPS=1
        HALVA_GRAD_ACCUM=1
        HALVA_EVAL_MAX_SAMPLES=2
        HALVA_AUX_MAX_SAMPLES=2
        HALVA_EVAL_SHARDS=1
        HALVA_GPU_LIST="${HALVA_GPU_LIST:-0}"
        HALVA_SKIP_AMBER_METRICS=True
        prepare
        train
        export_model
        evaluate
        echo "HALVA_SMOKE=PASS output=$OUT (AMBER inference checked; official AMBER score skipped for 2-row smoke)"
        ;;
    *) echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

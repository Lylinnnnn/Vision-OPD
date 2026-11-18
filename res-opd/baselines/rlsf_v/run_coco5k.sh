#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <qwen2b|qwen4b|qwen8b|internvl2b> <prepare|generate|convert|train|export|infer-smoke|eval>}"
PHASE="${2:?Specify a phase: prepare, preflight, generate, convert, train, sync-adapter, sync-merged, export, infer-smoke, eval}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
UPSTREAM="$RES_OPD_ROOT/third_party/baselines/RLSF-V"
HF_DIR="$UPSTREAM/train/hf_series"
WORK_ROOT="${RLSF_V_WORK_ROOT:-$RES_OPD_ROOT/checkpoints/rlsf_v_coco5k}"
mkdir -p "$WORK_ROOT"
WORK_ROOT="$(cd "$WORK_ROOT" && pwd)"
RUN_DIR="$WORK_ROOT/$MODEL"
DATA_ROOT="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"
QUESTIONS="$WORK_ROOT/questions.jsonl"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_SAMPLES="${RLSF_V_NUM_SAMPLES:-10}"
OSS_BASE="${RLSF_V_OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4/RLSF_V_COCO5K}"
OSS_RUN="$OSS_BASE/$(basename "$WORK_ROOT")/$MODEL"

sync_artifact() {
    local artifact="$1"
    local extra_args=()
    if [[ "${RLSF_V_KEEP_LOCAL_AFTER_SYNC:-0}" != "1" ]]; then
        extra_args+=(--delete-source-after-sync)
    elif [[ "$artifact" == adapter && "${RLSF_V_PRUNE_CHECKPOINTS_AFTER_SYNC:-1}" == "1" ]]; then
        extra_args+=(--prune-checkpoints)
    fi
    "$PYTHON_BIN" "$SCRIPT_DIR/sync_oss.py" upload \
        --source "$RUN_DIR/$artifact" --destination "$OSS_RUN/$artifact" \
        --work-root "$WORK_ROOT" --model "$MODEL" --artifact "$artifact" \
        "${extra_args[@]}"
}

restore_artifact() {
    local artifact="$1" target="${2:-$RUN_DIR/$1}"
    "$PYTHON_BIN" "$SCRIPT_DIR/sync_oss.py" restore \
        --source "$OSS_RUN/$artifact" --destination "$target" \
        --work-root "$WORK_ROOT" --model "$MODEL" --artifact "$artifact"
}

cleanup_artifact() {
    local artifact="$1" target="${2:-$RUN_DIR/$1}"
    "$PYTHON_BIN" "$SCRIPT_DIR/sync_oss.py" cleanup \
        --destination "$target" --work-root "$WORK_ROOT" \
        --model "$MODEL" --artifact "$artifact"
}

stage_merged() {
    STAGED_MERGED="$RUN_DIR/.merged_stage_$$"
    restore_artifact merged "$STAGED_MERGED"
    trap 'cleanup_artifact merged "$STAGED_MERGED"' EXIT
}

case "$MODEL" in
    qwen2b) SPEC=qwen3-8b; DEFAULT_MODEL_PATH=/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct ;;
    qwen4b) SPEC=qwen3-8b; DEFAULT_MODEL_PATH=/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-4B-Instruct ;;
    qwen8b) SPEC=qwen3-8b; DEFAULT_MODEL_PATH=/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-8B-Instruct ;;
    internvl2b) SPEC=internvl35-8b; DEFAULT_MODEL_PATH=/home/liuyanlin.lyl/notebook/model/internvl/InternVL3-2B-hf ;;
    *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
if [[ ! -s "$MODEL_PATH/config.json" ]]; then
    echo "Missing base model config: $MODEL_PATH/config.json" >&2
    exit 1
fi
MODEL_PATH="$(cd "$MODEL_PATH" && pwd)"
"$PYTHON_BIN" - "$MODEL_PATH/config.json" "$MODEL" <<'PY'
import json
import sys

config = json.load(open(sys.argv[1], encoding="utf-8"))
family = "internvl" if sys.argv[2] == "internvl2b" else "qwen3_vl"
if config.get("model_type") != family:
    raise SystemExit(f"Expected {family} base model, got {config.get('model_type')!r}")
PY
if [[ ! -d "$HF_DIR/generate" ]]; then
    echo "Missing vendored RLSF-V source: $HF_DIR" >&2
    exit 1
fi

case "$PHASE" in
    prepare)
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" prepare --model "$MODEL"
        "$PYTHON_BIN" "$SCRIPT_DIR/prepare_coco5k.py" \
            --train-parquet "$RES_OPD_ROOT/data/train_5k.parquet" \
            --test-json "$RES_OPD_ROOT/data/test_1000.json" \
            --data-root "$DATA_ROOT" --output "$QUESTIONS" \
            --max-samples "${RLSF_V_MAX_QUESTIONS:-0}"
        ;;
    preflight)
        test -s "$QUESTIONS" || { echo "Run prepare first: $QUESTIONS" >&2; exit 1; }
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" pipeline --model "$MODEL"
        cd "$HF_DIR"
        MODEL_PATH="$MODEL_PATH" QUESTION_FILE="$QUESTIONS" \
        OUTPUT_DIR="$RUN_DIR/generate" PYTHON_BIN="$PYTHON_BIN" DRY_RUN=1 \
        NUM_SAMPLES="$NUM_SAMPLES" BATCH_SIZE="${RLSF_V_BATCH_SIZE:-1}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        bash generate/generate_hf.sh "$SPEC"
        ;;
    generate)
        test -s "$QUESTIONS" || { echo "Run prepare first: $QUESTIONS" >&2; exit 1; }
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" generate --model "$MODEL"
        if [[ -e "$RUN_DIR/generate/answers.jsonl" || -e "$RUN_DIR/generate/logits_merged.pt" ]]; then
            echo "Generation outputs already exist; refusing to overwrite $RUN_DIR/generate" >&2
            exit 1
        fi
        cd "$HF_DIR"
        MODEL_PATH="$MODEL_PATH" QUESTION_FILE="$QUESTIONS" \
        OUTPUT_DIR="$RUN_DIR/generate" PYTHON_BIN="$PYTHON_BIN" \
        NUM_SAMPLES="$NUM_SAMPLES" BATCH_SIZE="${RLSF_V_BATCH_SIZE:-1}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
        bash generate/generate_hf.sh "$SPEC"
        "$PYTHON_BIN" "$SCRIPT_DIR/audit_generation.py" \
            --questions "$QUESTIONS" \
            --answers "$RUN_DIR/generate/answers.jsonl" \
            --logits "$RUN_DIR/generate/logits_merged.pt" \
            --num-samples "$NUM_SAMPLES"
        ;;
    convert)
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" convert --model "$MODEL"
        "$PYTHON_BIN" "$SCRIPT_DIR/audit_generation.py" \
            --questions "$QUESTIONS" \
            --answers "$RUN_DIR/generate/answers.jsonl" \
            --logits "$RUN_DIR/generate/logits_merged.pt" \
            --num-samples "$NUM_SAMPLES"
        if [[ -e "$RUN_DIR/generate/${MODEL}_dpo_data.json" ]]; then
            echo "DPO output already exists; refusing to overwrite it" >&2
            exit 1
        fi
        cd "$HF_DIR"
        MODEL_PATH="$MODEL_PATH" OUTPUT_DIR="$RUN_DIR/generate" \
        CONVERT_NAME="$MODEL" RUN_GENERATE=0 RUN_CONVERT=1 \
        MAX_WORKERS="${RLSF_V_CONVERT_WORKERS:-8}" PYTHON_BIN="$PYTHON_BIN" \
        bash generate/run_hf_dpo_pipeline.sh "$SPEC"
        ;;
    train)
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" train --model "$MODEL"
        if [[ "${RLSF_V_SYNC_OSS:-1}" == "1" ]] && \
            ossutil stat "$OSS_RUN/adapter/adapter_config.json" >/dev/null 2>&1; then
            echo "Adapter already exists on OSS; refusing to retrain over it: $OSS_RUN/adapter" >&2
            exit 1
        fi
        if [[ -e "$RUN_DIR/adapter/adapter_config.json" ]]; then
            echo "Adapter already exists; refusing to retrain over it" >&2
            exit 1
        fi
        "$PYTHON_BIN" "$SCRIPT_DIR/materialize.py" train \
            --model "$MODEL" --model-path "$MODEL_PATH" \
            --upstream "$UPSTREAM" --run-dir "$RUN_DIR" --questions "$QUESTIONS"
        cd "$UPSTREAM/train/hf_series/LlamaFactory"
        llamafactory-cli train "$RUN_DIR/train.yaml"
        test -s "$RUN_DIR/adapter/adapter_config.json" || {
            echo "Training returned success without an adapter checkpoint" >&2; exit 1;
        }
        if [[ "${RLSF_V_SYNC_OSS:-1}" == "1" ]]; then
            sync_artifact adapter
        else
            echo "WARNING: adapter remains local; RLSF_V_SYNC_OSS=0"
        fi
        ;;
    sync-adapter)
        test -s "$RUN_DIR/adapter/adapter_config.json" || {
            echo "No completed adapter checkpoint to sync" >&2; exit 1;
        }
        sync_artifact adapter
        ;;
    sync-merged)
        test -s "$RUN_DIR/merged/config.json" || {
            echo "No completed merged model to sync" >&2; exit 1;
        }
        sync_artifact merged
        ;;
    export)
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" export --model "$MODEL"
        STAGED_ADAPTER=0
        if [[ ! -e "$RUN_DIR/adapter" ]]; then
            restore_artifact adapter
            STAGED_ADAPTER=1
        fi
        cleanup_export() {
            if [[ "$STAGED_ADAPTER" == "1" ]]; then
                cleanup_artifact adapter
            fi
        }
        trap cleanup_export EXIT
        "$PYTHON_BIN" "$SCRIPT_DIR/materialize.py" export \
            --model "$MODEL" --model-path "$MODEL_PATH" \
            --upstream "$UPSTREAM" --run-dir "$RUN_DIR" --questions "$QUESTIONS"
        if [[ -e "$RUN_DIR/merged/config.json" ]]; then
            echo "Merged output already exists; refusing to overwrite it" >&2
            exit 1
        fi
        cd "$UPSTREAM/train/hf_series/LlamaFactory"
        llamafactory-cli export "$RUN_DIR/export.yaml"
        test -s "$RUN_DIR/merged/config.json" || {
            echo "Export returned success without merged config.json" >&2; exit 1;
        }
        if [[ "${RLSF_V_SYNC_MERGED_OSS:-1}" == "1" ]]; then
            sync_artifact merged
        fi
        ;;
    infer-smoke)
        "$PYTHON_BIN" "$SCRIPT_DIR/check_env.py" generate --model "$MODEL"
        test -s "$QUESTIONS" || { echo "Run prepare first" >&2; exit 1; }
        stage_merged
        SMOKE_DIR="$RUN_DIR/infer_smoke"
        if [[ -e "$SMOKE_DIR/answers.jsonl" ]]; then
            echo "Smoke answers already exist; refusing to overwrite $SMOKE_DIR" >&2
            exit 1
        fi
        mkdir -p "$SMOKE_DIR"
        "$PYTHON_BIN" - "$QUESTIONS" "$SMOKE_DIR/questions.jsonl" <<'PY'
import sys
from pathlib import Path

rows = Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()
if len(rows) < 2:
    raise SystemExit("Need at least two questions for inference smoke")
Path(sys.argv[2]).write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
PY
        if [[ "$MODEL" == internvl2b ]]; then
            INFER_SCRIPT="$HF_DIR/generate/internvl3_5_infer.py"
        else
            INFER_SCRIPT="$HF_DIR/generate/qwen3_infer.py"
        fi
        CUDA_VISIBLE_DEVICES="${RLSF_V_SMOKE_GPU:-0}" "$PYTHON_BIN" "$INFER_SCRIPT" \
            --model-name "$STAGED_MERGED" \
            --question-file "$SMOKE_DIR/questions.jsonl" \
            --answers-file "$SMOKE_DIR/answers.jsonl" \
            --num-chunks 1 --chunk-idx 0 --num-samples 1 --batch-size 1 \
            --temperature 0 --max-new-tokens "${RLSF_V_INFER_MAX_NEW_TOKENS:-128}"
        "$PYTHON_BIN" "$SCRIPT_DIR/audit_generation.py" \
            --questions "$SMOKE_DIR/questions.jsonl" \
            --answers "$SMOKE_DIR/answers.jsonl" \
            --logits "$SMOKE_DIR/all_logits_chunk0_merged.pt" \
            --num-samples 1
        echo "INFER_SMOKE=PASS answers=$SMOKE_DIR/answers.jsonl"
        ;;
    eval)
        stage_merged
        OUT="$RES_OPD_ROOT/eval_results/external_baselines/full/$(basename "$MODEL_PATH")/train5000_test1000_original_sr1p0/rlsf_v_coco5k_official_v2"
        cd "$VISION_OPD_ROOT"
        PYTHON_BIN="$PYTHON_BIN" RES_OPD_DATA_ROOT="$DATA_ROOT" \
        EVAL_OUTPUT_DIR="$OUT" DATASET_VERSION=full \
        DEGRADATION_MODE=original STUDENT_RATIO=1.0 \
        CHAIR_PROMPT='Describe the image.' CHAIR_MAX_SAMPLES=1000 \
        POPE_SOURCE=res-opd-test AMBER_EVAL_TYPE=a \
        EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}" \
        GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}" \
        VLLM_BASE_PORT="${VLLM_BASE_PORT:-34100}" \
        VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-3600}" \
        bash "$RES_OPD_ROOT/scripts/eval_after_merge_sharded_8gpu.sh" \
            "$STAGED_MERGED" "$MODEL" chair,pope,amber
        ;;
    *) echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

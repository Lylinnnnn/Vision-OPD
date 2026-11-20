#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?Usage: $0 <qwen2b|qwen4b|qwen8b|internvl2b> <prepare|preflight|generate|train|export|eval|smoke>}"
PHASE="${2:?Specify phase}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
RUN_NAME="${APASI_RUN_NAME:-apasi_base_coco5k}"
WORK_ROOT="${APASI_WORK_ROOT:-$ROOT/checkpoints/apasi_coco5k}"
EVAL_WORK_ROOT="${APASI_EVAL_WORK_ROOT:-/tmp/apasi_coco5k}"
DATA_ROOT="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"
UPSTREAM="$ROOT/third_party/baselines/APASI"
LF_ROOT="$ROOT/third_party/baselines/RLSF-V/train/hf_series/LlamaFactory"
GEN_PY="${APASI_GENERATION_PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
TRAIN_PY="${APASI_TRAIN_PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/rlsf-v-lf/bin/python3}"
EVAL_PY="${APASI_EVAL_PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
OSS_BASE="${APASI_OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4/APASI_COCO5K}"

case "$MODEL" in
    qwen2b) MODEL_PATH="${APASI_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}" ;;
    qwen4b) MODEL_PATH="${APASI_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-4B-Instruct}" ;;
    qwen8b) MODEL_PATH="${APASI_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-8B-Instruct}" ;;
    internvl2b) MODEL_PATH="${APASI_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/internvl/InternVL3-2B-hf}" ;;
    *) echo "Unsupported model: $MODEL" >&2; exit 2 ;;
esac
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "Invalid APASI_RUN_NAME" >&2; exit 2; }
test -s "$MODEL_PATH/config.json" || { echo "Missing model config: $MODEL_PATH" >&2; exit 1; }
MODEL_DIR="$WORK_ROOT/$RUN_NAME/$MODEL"
QUESTIONS="$MODEL_DIR/questions.jsonl"
SHARDS="${APASI_GENERATION_SHARDS:-8}"
GPU_LIST="${APASI_GPU_LIST:-0,1,2,3,4,5,6,7}"
IFS=',' read -ra GPUS <<< "$GPU_LIST"
if [[ "$PHASE" == generate || "$PHASE" == eval ]] && [[ "$SHARDS" -ne "${#GPUS[@]}" ]]; then
    echo "APASI_GENERATION_SHARDS must equal the number of APASI_GPU_LIST entries" >&2
    exit 2
fi

storage() {
    "$TRAIN_PY" "$SCRIPT_DIR/storage.py" "$1" --model "$MODEL" \
        --artifact "$2" --run-name "$RUN_NAME" --work-root "$3" "${@:4}"
}

prepare() {
    test ! -e "$QUESTIONS" || { echo "Questions already exist: $QUESTIONS" >&2; exit 1; }
    mkdir -p "$MODEL_DIR"
    "$GEN_PY" "$ROOT/baselines/rlsf_v/prepare_coco5k.py" \
        --train-parquet "$ROOT/data/train_5k.parquet" \
        --test-json "$ROOT/data/test_1000.json" --data-root "$DATA_ROOT" \
        --output "$QUESTIONS" --max-samples "${APASI_MAX_QUESTIONS:-0}"
    if [[ "${APASI_MAX_QUESTIONS:-0}" == "0" ]]; then
        test "$(wc -l < "$QUESTIONS")" -eq 5000 || {
            echo "Full APASI run requires exactly 5000 COCO training images" >&2; exit 1;
        }
    fi
}

preflight() {
    test -s "$QUESTIONS" || { echo "Run prepare first: $QUESTIONS" >&2; exit 1; }
    "$GEN_PY" - "$MODEL_PATH/config.json" "$MODEL" <<'PY'
import json, sys
from importlib import import_module
config = json.load(open(sys.argv[1], encoding="utf-8"))
expected = "internvl" if sys.argv[2] == "internvl2b" else "qwen3_vl"
assert config.get("model_type") == expected, (config.get("model_type"), expected)
for package in ("torch", "transformers", "PIL", "nltk", "pandas", "pyarrow"):
    import_module(package)
import nltk
for resource in ("tokenizers/punkt_tab", "taggers/averaged_perceptron_tagger_eng", "corpora/wordnet"):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.data.find(resource + ".zip")
print("APASI_GENERATION_PREFLIGHT=PASS")
PY
    bash "$SCRIPT_DIR/fetch_upstream.sh"
}

generate_stage() {
    local stage="$1" input="$2" output_dir="$3" max_tokens="$4"
    local pids=() failed=0 i gpu pid
    mkdir -p "$output_dir" "$MODEL_DIR/logs"
    for ((i=0; i<SHARDS; i++)); do
        gpu="${GPUS[$i]}"
        CUDA_VISIBLE_DEVICES="$gpu" "$GEN_PY" "$SCRIPT_DIR/generate.py" "$stage" \
            --model "$MODEL" --model-path "$MODEL_PATH" --input "$input" \
            --output "$output_dir/shard_$(printf '%02d' "$i").jsonl" \
            --shard-count "$SHARDS" --shard-index "$i" \
            --max-new-tokens "$max_tokens" \
            > "$MODEL_DIR/logs/${stage}_$(printf '%02d' "$i").log" 2>&1 &
        pid="$!"
        pids+=("$pid")
        echo "APASI_${stage} shard=$i gpu=$gpu pid=$pid"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || failed=1
    done
    if [[ "$failed" -ne 0 ]]; then
        echo "APASI $stage failed; inspect $MODEL_DIR/logs/${stage}_*.log. Completed rows are resumable." >&2
        return 1
    fi
}

generate() {
    preflight
    generate_stage preferred "$QUESTIONS" "$MODEL_DIR/preferred" \
        "${APASI_PREFERRED_MAX_NEW_TOKENS:-512}"
    if [[ ! -s "$MODEL_DIR/plans.jsonl" ]]; then
        "$GEN_PY" "$SCRIPT_DIR/prepare.py" --questions "$QUESTIONS" \
            --preferred-dir "$MODEL_DIR/preferred" --work-dir "$MODEL_DIR" \
            --upstream "$UPSTREAM" --injection-rate "${APASI_INJECTION_RATE:-0.2}"
    fi
    generate_stage inject "$MODEL_DIR/plans.jsonl" "$MODEL_DIR/injected" \
        "${APASI_BLIND_MAX_NEW_TOKENS:-64}"
    echo "APASI_GENERATE=PASS plans=$(wc -l < "$MODEL_DIR/plans.jsonl")"
}

train() {
    test -s "$MODEL_DIR/plans.jsonl" || { echo "Run generate first" >&2; exit 1; }
    test ! -e "$MODEL_DIR/adapter" || { echo "Adapter already exists: $MODEL_DIR/adapter" >&2; exit 1; }
    command -v llamafactory-cli >/dev/null || { echo "Activate rlsf-v-lf first" >&2; exit 1; }
    command -v ossutil >/dev/null || { echo "ossutil is required for verified checkpoint upload" >&2; exit 1; }
    test -s "$LF_ROOT/qwen3.yaml" || { echo "Missing LlamaFactory source: $LF_ROOT" >&2; exit 1; }
    if ossutil stat "$OSS_BASE/$RUN_NAME/$MODEL/adapter/adapter_config.json" >/dev/null 2>&1; then
        echo "Adapter already exists on OSS; refusing to retrain" >&2
        exit 1
    fi
    "$TRAIN_PY" "$SCRIPT_DIR/materialize.py" train --model "$MODEL" \
        --model-path "$MODEL_PATH" --lf-root "$LF_ROOT" \
        --work-dir "$MODEL_DIR" --questions "$QUESTIONS" \
        --epochs "${APASI_EPOCHS:-3}" --cutoff-len "${APASI_CUTOFF_LEN:-4096}" \
        --max-steps "${APASI_MAX_STEPS:-0}"
    llamafactory-cli train "$MODEL_DIR/train.yaml"
    test -s "$MODEL_DIR/adapter/adapter_config.json"
    storage upload adapter "$WORK_ROOT"
}

export_model() {
    command -v llamafactory-cli >/dev/null || { echo "Activate rlsf-v-lf first" >&2; exit 1; }
    command -v ossutil >/dev/null || { echo "ossutil is required for checkpoint restore and upload" >&2; exit 1; }
    test ! -e "$MODEL_DIR/merged" || { echo "Merged model already exists" >&2; exit 1; }
    local restored=0
    if [[ ! -s "$MODEL_DIR/adapter/adapter_config.json" ]]; then
        storage restore adapter "$WORK_ROOT"
        restored=1
    fi
    "$TRAIN_PY" "$SCRIPT_DIR/materialize.py" export --model "$MODEL" \
        --model-path "$MODEL_PATH" --lf-root "$LF_ROOT" \
        --work-dir "$MODEL_DIR" --questions "$QUESTIONS"
    llamafactory-cli export "$MODEL_DIR/export.yaml"
    test -s "$MODEL_DIR/merged/config.json"
    storage upload merged "$WORK_ROOT"
    if [[ "$restored" == 1 ]]; then
        storage cleanup adapter "$WORK_ROOT"
    fi
}

choose_port() {
    local requested="${APASI_BASE_PORT:-37500}" attempt i port busy
    command -v ss >/dev/null || { echo "ss is required for port preflight" >&2; exit 1; }
    for ((attempt=0; attempt<40; attempt++)); do
        local base=$((requested + 100 * attempt))
        busy=0
        for ((i=0; i<${#GPUS[@]}; i++)); do
            port=$((base + 10 * i))
            if [[ -n "$(ss -H -ltn "( sport = :$port )")" ]]; then busy=1; break; fi
        done
        if [[ "$busy" == 0 ]]; then echo "$base"; return; fi
    done
    echo "No free eight-shard port block from $requested" >&2
    return 1
}

evaluate() {
    command -v ossutil >/dev/null || { echo "ossutil is required for merged-model restore" >&2; exit 1; }
    local staged_name="$$" staged="$EVAL_WORK_ROOT/$RUN_NAME/$MODEL/.merged_stage_$$"
    local output="$ROOT/eval_results/external_baselines/full/$(basename "$MODEL_PATH")/train5000_test1000_original_sr1p0/${RUN_NAME}_official_v2"
    if [[ -e "$output/chair_metrics.json" && "${APASI_ALLOW_EVAL_OVERWRITE:-0}" != 1 ]]; then
        echo "CHAIR metrics already exist; refusing to overwrite: $output" >&2; exit 1
    fi
    storage restore merged "$EVAL_WORK_ROOT" --stage "$staged_name"
    local port
    port="$(choose_port)"
    cd "$REPO"
    PYTHON_BIN="$EVAL_PY" RES_OPD_DATA_ROOT="$DATA_ROOT" EVAL_OUTPUT_DIR="$output" \
    DATASET_VERSION=full DEGRADATION_MODE=original STUDENT_RATIO=1.0 \
    CHAIR_PROMPT='Describe the image.' CHAIR_MAX_SAMPLES="${APASI_EVAL_MAX_SAMPLES:-1000}" \
    POPE_SOURCE=res-opd-test POPE_MAX_SAMPLES="${APASI_AUX_MAX_SAMPLES:-0}" \
    AMBER_ROOT="$DATA_ROOT/AMBER" AMBER_EVAL_TYPE=a \
    AMBER_MAX_SAMPLES="${APASI_AUX_MAX_SAMPLES:-0}" \
    AMBER_SKIP_OFFICIAL_EVAL="${APASI_SKIP_AMBER_METRICS:-False}" \
    EVAL_SHARD_COUNT="${#GPUS[@]}" GPU_LIST="$GPU_LIST" \
    VLLM_BASE_PORT="$port" VLLM_PORT_CLEANUP=False \
    VLLM_STARTUP_TIMEOUT="${APASI_VLLM_TIMEOUT:-3600}" \
    bash "$ROOT/scripts/eval_after_merge_sharded_8gpu.sh" "$staged" 0 instruct chair,pope,amber
    storage cleanup merged "$EVAL_WORK_ROOT" --stage "$staged_name"
    echo "APASI_EVAL=PASS output=$output"
}

case "$PHASE" in
    prepare) prepare ;;
    preflight) preflight ;;
    generate) generate ;;
    train) train ;;
    export) export_model ;;
    eval) evaluate ;;
    smoke)
        [[ -n "${APASI_RUN_NAME:-}" ]] || { echo "Set a unique APASI_RUN_NAME for smoke" >&2; exit 2; }
        [[ "$GPU_LIST" != *,* ]] || { echo "Smoke requires a single APASI_GPU_LIST device" >&2; exit 2; }
        export APASI_WORK_ROOT="${APASI_WORK_ROOT:-$ROOT/checkpoints/apasi_smoke}"
        export APASI_EVAL_WORK_ROOT="${APASI_EVAL_WORK_ROOT:-/tmp/apasi_smoke}"
        export APASI_MAX_QUESTIONS="${APASI_MAX_QUESTIONS:-64}"
        export APASI_GENERATION_SHARDS=1 APASI_GPU_LIST="${APASI_GPU_LIST:-0}"
        export CUDA_VISIBLE_DEVICES="$APASI_GPU_LIST"
        export APASI_MAX_STEPS=1 APASI_EPOCHS=1
        export APASI_EVAL_MAX_SAMPLES=2 APASI_AUX_MAX_SAMPLES=2
        export APASI_SKIP_AMBER_METRICS=True
        bash "$SCRIPT_DIR/run_coco5k.sh" "$MODEL" prepare
        bash "$SCRIPT_DIR/run_coco5k.sh" "$MODEL" generate
        bash "$SCRIPT_DIR/run_coco5k.sh" "$MODEL" train
        bash "$SCRIPT_DIR/run_coco5k.sh" "$MODEL" export
        bash "$SCRIPT_DIR/run_coco5k.sh" "$MODEL" eval
        echo "APASI_SMOKE=PASS model=$MODEL run=$RUN_NAME"
        ;;
    *) echo "Unsupported phase: $PHASE" >&2; exit 2 ;;
esac

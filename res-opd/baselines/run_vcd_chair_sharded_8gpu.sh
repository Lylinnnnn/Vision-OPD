#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
source "${RES_OPD_ROOT}/scripts/path_utils.sh"

MODEL_PATH="${1:?Usage: $0 <Qwen3VL model path> [output directory]}"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"

MODEL_NAME="$(basename "$MODEL_PATH")"
TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
VCD_OUTPUT_SUFFIX="${VCD_OUTPUT_SUFFIX:-chair_external_vcd_official}"
FINAL_OUTPUT_DIR="${2:-${RES_OPD_ROOT}/eval_results/external_baselines/full/${MODEL_NAME}/train5000_test1000_original_sr1p0/${VCD_OUTPUT_SUFFIX}}"
RUN_ID="${VCD_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SHARD_ROOT="${VCD_SHARD_ROOT:-${FINAL_OUTPUT_DIR}/.shards/${RUN_ID}}"
VCD_ALPHA="${VCD_ALPHA:-1.0}"
VCD_BETA="${VCD_BETA:-0.1}"
VCD_NOISE_STEP="${VCD_NOISE_STEP:-500}"
VCD_DECODE_MODE="${VCD_DECODE_MODE:-sample}"
VCD_TEMPERATURE="${VCD_TEMPERATURE:-1.0}"
VCD_TOP_P="${VCD_TOP_P:-1.0}"
VCD_TOP_K="${VCD_TOP_K:-0}"
VCD_SEED="${VCD_SEED:-42}"
VCD_MAX_NEW_TOKENS="${VCD_MAX_NEW_TOKENS:-8192}"
VCD_MAX_SAMPLES="${VCD_MAX_SAMPLES:-0}"
VCD_TORCH_DTYPE="${VCD_TORCH_DTYPE:-bfloat16}"
VCD_SHARD_START_DELAY="${VCD_SHARD_START_DELAY:-2}"
VCD_KEEP_SHARDS="${VCD_KEEP_SHARDS:-False}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Describe the image.}"

[[ -d "$MODEL_PATH" ]] || { echo "ERROR: model path does not exist: $MODEL_PATH" >&2; exit 1; }
[[ -f "$TEST_JSON" ]] || { echo "ERROR: test JSON does not exist: $TEST_JSON" >&2; exit 1; }
[[ -f "${RES_OPD_ROOT}/third_party/baselines/VCD/vcd_utils/vcd_add_noise.py" ]] || {
    echo "ERROR: VCD source is not initialized. Run git submodule update --init --recursive." >&2
    exit 1
}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt "$EVAL_SHARD_COUNT" ]]; then
    echo "ERROR: GPU_LIST has ${#GPUS[@]} GPUs but EVAL_SHARD_COUNT=${EVAL_SHARD_COUNT}" >&2
    exit 1
fi

mkdir -p "$SHARD_ROOT" "$FINAL_OUTPUT_DIR"
echo "============================================================"
echo " Qwen3VL VCD Official CHAIR Evaluation"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "Output:      $FINAL_OUTPUT_DIR"
echo "Prompt:      $CHAIR_PROMPT"
echo "Shards:      $EVAL_SHARD_COUNT on $GPU_LIST"
echo "VCD:         alpha=$VCD_ALPHA beta=$VCD_BETA noise=$VCD_NOISE_STEP"
echo "Decode:      $VCD_DECODE_MODE temp=$VCD_TEMPERATURE top_p=$VCD_TOP_P top_k=$VCD_TOP_K"
echo "Max tokens:  $VCD_MAX_NEW_TOKENS"
echo "Max samples: $VCD_MAX_SAMPLES"
echo "============================================================"

pids=()
shard_dirs=()
cleanup_children() {
    local pid
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_children INT TERM

for shard_index in $(seq 0 $((EVAL_SHARD_COUNT - 1))); do
    gpu="${GPUS[$shard_index]}"
    shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "$shard_index")"
    log_file="${shard_dir}/eval.log"
    exit_file="${shard_dir}/exit_code"
    mkdir -p "${shard_dir}/out"
    rm -f "$exit_file"
    (
        set +e
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${SCRIPT_DIR}/eval_vcd_chair.py" \
            --model-path "$MODEL_PATH" \
            --test-json "$TEST_JSON" \
            --output-dir "${shard_dir}/out" \
            --prompt "$CHAIR_PROMPT" \
            --max-new-tokens "$VCD_MAX_NEW_TOKENS" \
            --max-samples "$VCD_MAX_SAMPLES" \
            --shard-count "$EVAL_SHARD_COUNT" \
            --shard-index "$shard_index" \
            --alpha "$VCD_ALPHA" \
            --beta "$VCD_BETA" \
            --noise-step "$VCD_NOISE_STEP" \
            --decode-mode "$VCD_DECODE_MODE" \
            --temperature "$VCD_TEMPERATURE" \
            --top-p "$VCD_TOP_P" \
            --top-k "$VCD_TOP_K" \
            --seed "$VCD_SEED" \
            --torch-dtype "$VCD_TORCH_DTYPE" \
            > "$log_file" 2>&1
        status=$?
        echo "$status" > "$exit_file"
        exit "$status"
    ) &
    pids+=("$!")
    shard_dirs+=("$shard_dir")
    echo "Started shard $shard_index on GPU $gpu: $log_file"
    sleep "$VCD_SHARD_START_DELAY"
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        failed=1
        echo "ERROR: shard $index failed; tail follows" >&2
        tail -n 100 "${shard_dirs[$index]}/eval.log" >&2 || true
    fi
done
trap - INT TERM

if [[ "$failed" -ne 0 ]]; then
    echo "ERROR: VCD evaluation failed. Shards retained at: $SHARD_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/merge_sharded_eval.py" \
    --final-output-dir "$FINAL_OUTPUT_DIR" \
    --shard-root "$SHARD_ROOT" \
    --eval-mode chair \
    --experiment-name "external-vcd-${MODEL_NAME}" \
    --seed "$VCD_SEED"

cp "${SHARD_ROOT}/shard_00/out/vcd_config.json" "${FINAL_OUTPUT_DIR}/vcd_config.json"
if ! res_opd_is_truthy "$VCD_KEEP_SHARDS"; then
    rm -rf "$SHARD_ROOT"
    rmdir "$(dirname "$SHARD_ROOT")" 2>/dev/null || true
else
    echo "Keeping shard outputs at: $SHARD_ROOT"
fi

echo "VCD official CHAIR evaluation complete: $FINAL_OUTPUT_DIR"

#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${RES_OPD_ROOT}/scripts/path_utils.sh"

MODEL_PATH="${1:?Usage: $0 <Qwen3VL model path> [output directory]}"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"

MODEL_NAME="$(basename "$MODEL_PATH")"
TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
OPERA_OUTPUT_SUFFIX="${OPERA_OUTPUT_SUFFIX:-chair_external_opera_official}"
FINAL_OUTPUT_DIR="${2:-${RES_OPD_ROOT}/eval_results/external_baselines/full/${MODEL_NAME}/train5000_test1000_original_sr1p0/${OPERA_OUTPUT_SUFFIX}}"
RUN_ID="${OPERA_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SHARD_ROOT="${OPERA_SHARD_ROOT:-${FINAL_OUTPUT_DIR}/.shards/${RUN_ID}}"
OPERA_NUM_BEAMS="${OPERA_NUM_BEAMS:-5}"
OPERA_SCALE_FACTOR="${OPERA_SCALE_FACTOR:-50.0}"
OPERA_THRESHOLD="${OPERA_THRESHOLD:-15}"
OPERA_NUM_ATTN_CANDIDATES="${OPERA_NUM_ATTN_CANDIDATES:-5}"
OPERA_PENALTY_WEIGHTS="${OPERA_PENALTY_WEIGHTS:-1.0}"
OPERA_WINDOW_SIZE="${OPERA_WINDOW_SIZE:-512}"
OPERA_MAX_NEW_TOKENS="${OPERA_MAX_NEW_TOKENS:-8192}"
OPERA_MAX_SAMPLES="${OPERA_MAX_SAMPLES:-0}"
OPERA_TORCH_DTYPE="${OPERA_TORCH_DTYPE:-bfloat16}"
OPERA_SHARD_START_DELAY="${OPERA_SHARD_START_DELAY:-3}"
OPERA_KEEP_SHARDS="${OPERA_KEEP_SHARDS:-False}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Describe the image.}"

[[ -d "$MODEL_PATH" ]] || { echo "ERROR: model path does not exist: $MODEL_PATH" >&2; exit 1; }
[[ -f "$TEST_JSON" ]] || { echo "ERROR: test JSON does not exist: $TEST_JSON" >&2; exit 1; }
[[ -f "${RES_OPD_ROOT}/third_party/baselines/OPERA/LICENSE" ]] || {
    echo "ERROR: OPERA source is not initialized. Run git submodule update --init --recursive." >&2
    exit 1
}

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt "$EVAL_SHARD_COUNT" ]]; then
    echo "ERROR: GPU_LIST has ${#GPUS[@]} GPUs but EVAL_SHARD_COUNT=${EVAL_SHARD_COUNT}" >&2
    exit 1
fi

mkdir -p "$SHARD_ROOT" "$FINAL_OUTPUT_DIR"
echo "============================================================"
echo " Qwen3VL OPERA Official CHAIR Evaluation"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "Output:      $FINAL_OUTPUT_DIR"
echo "Prompt:      $CHAIR_PROMPT"
echo "Shards:      $EVAL_SHARD_COUNT on $GPU_LIST"
echo "Beam:        $OPERA_NUM_BEAMS"
echo "OPERA:       scale=$OPERA_SCALE_FACTOR threshold=$OPERA_THRESHOLD candidates=$OPERA_NUM_ATTN_CANDIDATES"
echo "              penalty=$OPERA_PENALTY_WEIGHTS window=$OPERA_WINDOW_SIZE"
echo "Max tokens:  $OPERA_MAX_NEW_TOKENS"
echo "Max samples: $OPERA_MAX_SAMPLES"
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
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${SCRIPT_DIR}/eval_opera_chair.py" \
            --model-path "$MODEL_PATH" \
            --test-json "$TEST_JSON" \
            --output-dir "${shard_dir}/out" \
            --prompt "$CHAIR_PROMPT" \
            --max-new-tokens "$OPERA_MAX_NEW_TOKENS" \
            --max-samples "$OPERA_MAX_SAMPLES" \
            --shard-count "$EVAL_SHARD_COUNT" \
            --shard-index "$shard_index" \
            --num-beams "$OPERA_NUM_BEAMS" \
            --scale-factor "$OPERA_SCALE_FACTOR" \
            --threshold "$OPERA_THRESHOLD" \
            --num-attn-candidates "$OPERA_NUM_ATTN_CANDIDATES" \
            --penalty-weights "$OPERA_PENALTY_WEIGHTS" \
            --window-size "$OPERA_WINDOW_SIZE" \
            --torch-dtype "$OPERA_TORCH_DTYPE" \
            > "$log_file" 2>&1
        status=$?
        echo "$status" > "$exit_file"
        exit "$status"
    ) &
    pids+=("$!")
    shard_dirs+=("$shard_dir")
    echo "Started shard $shard_index on GPU $gpu: $log_file"
    sleep "$OPERA_SHARD_START_DELAY"
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        failed=1
        echo "ERROR: shard $index failed; tail follows" >&2
        tail -n 120 "${shard_dirs[$index]}/eval.log" >&2 || true
    fi
done
trap - INT TERM

if [[ "$failed" -ne 0 ]]; then
    echo "ERROR: OPERA evaluation failed. Shards retained at: $SHARD_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/merge_sharded_eval.py" \
    --final-output-dir "$FINAL_OUTPUT_DIR" \
    --shard-root "$SHARD_ROOT" \
    --eval-mode chair \
    --experiment-name "external-opera-${MODEL_NAME}" \
    --seed 42

cp "${SHARD_ROOT}/shard_00/out/opera_config.json" "${FINAL_OUTPUT_DIR}/opera_config.json"
if ! res_opd_is_truthy "$OPERA_KEEP_SHARDS"; then
    rm -rf "$SHARD_ROOT"
    rmdir "$(dirname "$SHARD_ROOT")" 2>/dev/null || true
else
    echo "Keeping shard outputs at: $SHARD_ROOT"
fi

echo "OPERA official CHAIR evaluation complete: $FINAL_OUTPUT_DIR"

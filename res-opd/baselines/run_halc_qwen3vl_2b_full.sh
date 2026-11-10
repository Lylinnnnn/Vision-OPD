#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
source "$RES_OPD_ROOT/scripts/path_utils.sh"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
res_opd_validate_python_bin "$PYTHON_BIN"
MODEL_PATH="${MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}"
DINO_CONFIG="${DINO_CONFIG:-$RES_OPD_ROOT/third_party/baselines/HALC/decoder_zoo/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py}"
DINO_WEIGHTS="${DINO_WEIGHTS:-/home/liuyanlin.lyl/notebook/model/groundingdino/groundingdino_swint_ogc.pth}"
BLIP_MODEL="${BLIP_MODEL:-/home/liuyanlin.lyl/notebook/model/blip/blip-itm-base-coco}"
RES_OPD_DATA_ROOT="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
HALC_MIN_FREE_MIB="${HALC_MIN_FREE_MIB:-36000}"
HALC_AUX_BASE_PORT="${HALC_AUX_BASE_PORT:-29700}"
HALC_SMOKE_PORT="${HALC_SMOKE_PORT:-29600}"
EVAL_API_REQUEST_TIMEOUT_SECONDS="${EVAL_API_REQUEST_TIMEOUT_SECONDS:-1800}"
OUTPUT_DIR="${OUTPUT_DIR:-$RES_OPD_ROOT/eval_results/external_baselines/full/Qwen3VL-2B-Instruct/train5000_test1000_original_sr1p0/chair_external_halc_qwen3vl_adapted_2048}"

export PYTHON_BIN RES_OPD_DATA_ROOT EVAL_API_REQUEST_TIMEOUT_SECONDS
TORCH_LIB="$("$PYTHON_BIN" -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} != 8 )); then
    echo "ERROR: HALC 2B requires eight distinct GPUs; got $GPU_LIST" >&2
    exit 1
fi
if [[ ! "$HALC_MIN_FREE_MIB" =~ ^[0-9]+$ ]]; then
    echo "ERROR: HALC_MIN_FREE_MIB must be a nonnegative integer" >&2
    exit 1
fi
mapfile -t FREE_MIB < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
declare -A SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
    if [[ ! "$gpu" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid GPU index: $gpu" >&2
        exit 1
    fi
    if [[ -n "${SEEN_GPUS[$gpu]:-}" ]]; then
        echo "ERROR: Invalid or duplicate GPU index: $gpu" >&2
        exit 1
    fi
    SEEN_GPUS[$gpu]=1
    free="${FREE_MIB[$gpu]:-}"
    if [[ ! "$free" =~ ^[0-9]+$ ]] || (( free < HALC_MIN_FREE_MIB )); then
        echo "ERROR: GPU $gpu has ${free:-unknown} MiB free; need $HALC_MIN_FREE_MIB MiB" >&2
        exit 1
    fi
done

cd "$ROOT"
echo "HALC_2B model=$MODEL_PATH data=$RES_OPD_DATA_ROOT output=$OUTPUT_DIR"
"$PYTHON_BIN" "$SCRIPT_DIR/eval_halc_chair.py" \
    --model-path "$MODEL_PATH" --output-dir "$OUTPUT_DIR/.smoke_chair_2048" \
    --detector-config "$DINO_CONFIG" --detector-weights "$DINO_WEIGHTS" \
    --blip-model-path "$BLIP_MODEL" --max-samples 5 --max-new-tokens 2048 \
    --require-trigger --require-change

HALC_ENABLE_FULL_RUN=1 HALC_MAX_NEW_TOKENS=2048 HALC_MAX_SAMPLES=0 \
    HALC_RUN_ID=qwen3vl2b_2048_v1 HALC_SHARD_COUNT=8 HALC_SHARD_START_DELAY=5 \
    GPU_LIST="$GPU_LIST" TEST_JSON="$RES_OPD_ROOT/data/test_1000.json" \
    CHAIR_PROMPT='Describe the image.' \
    bash "$SCRIPT_DIR/run_halc_chair_sharded_8gpu.sh" \
    "$MODEL_PATH" "$DINO_CONFIG" "$DINO_WEIGHTS" "$BLIP_MODEL" "$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/smoke_halc_aux.py" \
    --model-path "$MODEL_PATH" --halc-dir "$OUTPUT_DIR" \
    --detector-config "$DINO_CONFIG" --data-root "$RES_OPD_DATA_ROOT" \
    --gpu "${GPUS[0]}" --port "$HALC_SMOKE_PORT" --samples 10

"$PYTHON_BIN" "$SCRIPT_DIR/run_external_aux_benchmarks.py" \
    --model-path "$MODEL_PATH" --methods halc --benchmarks pope,amber \
    --halc-dir "$OUTPUT_DIR" --detector-config "$DINO_CONFIG" \
    --data-root "$RES_OPD_DATA_ROOT" --gpu-list "$GPU_LIST" \
    --base-port "$HALC_AUX_BASE_PORT" --min-free-mib "$HALC_MIN_FREE_MIB" \
    --startup-timeout 1800

echo "HALC_2B_FULL=PASS output=$OUTPUT_DIR"

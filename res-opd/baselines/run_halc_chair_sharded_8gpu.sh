#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${RES_OPD_ROOT}/scripts/path_utils.sh"

MODEL_PATH="${1:?Usage: $0 <model path> <GroundingDINO config> <GroundingDINO weights> <BLIP ITM model path> [output directory]}"
DINO_CONFIG="${2:?GroundingDINO config is required}"
DINO_WEIGHTS="${3:?GroundingDINO weights are required}"
BLIP_MODEL="${4:?BLIP ITM model path is required}"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
res_opd_validate_python_bin "$PYTHON_BIN"

MODEL_NAME="$(basename "$MODEL_PATH")"
MODEL_FAMILY="${MODEL_FAMILY:-qwen3vl}"
[[ "$MODEL_FAMILY" == qwen3vl || "$MODEL_FAMILY" == internvl3 ]] || { echo "ERROR: invalid MODEL_FAMILY" >&2; exit 1; }
TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
HALC_SHARD_COUNT="${HALC_SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
HALC_MAX_SAMPLES="${HALC_MAX_SAMPLES:-0}"
HALC_MAX_NEW_TOKENS="${HALC_MAX_NEW_TOKENS:-8192}"
HALC_SHARD_START_DELAY="${HALC_SHARD_START_DELAY:-3}"
CHAIR_PROMPT="${CHAIR_PROMPT:-Describe the image.}"
FINAL_OUTPUT_DIR="${5:-${RES_OPD_ROOT}/eval_results/external_baselines/full/${MODEL_NAME}/train5000_test1000_original_sr1p0/chair_external_halc_${MODEL_FAMILY}_adapted_official}"
RUN_ID="${HALC_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SHARD_ROOT="${HALC_SHARD_ROOT:-${FINAL_OUTPUT_DIR}/.shards/${RUN_ID}}"

[[ "${HALC_ENABLE_FULL_RUN:-0}" == 1 ]] || {
    echo "ERROR: HALC adapter is not validated against upstream; set HALC_ENABLE_FULL_RUN=1 after GPU smoke." >&2
    exit 1
}
[[ -f "$TEST_JSON" ]] || { echo "ERROR: test JSON not found: $TEST_JSON" >&2; exit 1; }
[[ -d "$MODEL_PATH" && -f "$DINO_CONFIG" && -s "$DINO_WEIGHTS" && -d "$BLIP_MODEL" ]] || {
    echo "ERROR: model, GroundingDINO, or BLIP paths are missing" >&2
    exit 1
}
[[ "$HALC_SHARD_COUNT" =~ ^[1-9][0-9]*$ && "$HALC_MAX_SAMPLES" =~ ^[0-9]+$ ]] || {
    echo "ERROR: invalid shard count or max samples" >&2
    exit 1
}
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if (( ${#GPUS[@]} < HALC_SHARD_COUNT )); then
    echo "ERROR: GPU_LIST has ${#GPUS[@]} GPUs but HALC_SHARD_COUNT=$HALC_SHARD_COUNT" >&2
    exit 1
fi

mkdir -p "$SHARD_ROOT" "$FINAL_OUTPUT_DIR"
echo "HALC/$MODEL_FAMILY adapted CHAIR: model=$MODEL_PATH output=$FINAL_OUTPUT_DIR shards=$HALC_SHARD_COUNT samples=$HALC_MAX_SAMPLES"

pids=()
shard_dirs=()
cleanup_children() {
    for pid in "${pids[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
}
trap cleanup_children INT TERM

for ((shard_index=0; shard_index<HALC_SHARD_COUNT; shard_index++)); do
    gpu="${GPUS[$shard_index]}"
    shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "$shard_index")"
    mkdir -p "${shard_dir}/out"
    (
        set +e
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" "${SCRIPT_DIR}/eval_halc_chair.py" \
            --model-family "$MODEL_FAMILY" \
            --model-path "$MODEL_PATH" \
            --test-json "$TEST_JSON" \
            --output-dir "${shard_dir}/out" \
            --detector-config "$DINO_CONFIG" \
            --detector-weights "$DINO_WEIGHTS" \
            --blip-model-path "$BLIP_MODEL" \
            --prompt "$CHAIR_PROMPT" \
            --max-samples "$HALC_MAX_SAMPLES" \
            --max-new-tokens "$HALC_MAX_NEW_TOKENS" \
            --shard-count "$HALC_SHARD_COUNT" \
            --shard-index "$shard_index" \
            --allow-unvalidated-full-run \
            > "${shard_dir}/eval.log" 2>&1
        status=$?
        echo "$status" > "${shard_dir}/exit_code"
        exit "$status"
    ) &
    pids+=("$!")
    shard_dirs+=("$shard_dir")
    echo "Started shard $shard_index on GPU $gpu: ${shard_dir}/eval.log"
    sleep "$HALC_SHARD_START_DELAY"
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
        failed=1
        echo "ERROR: shard $index failed; tail follows" >&2
        tail -n 80 "${shard_dirs[$index]}/eval.log" >&2 || true
    fi
done
trap - INT TERM
if (( failed )); then
    echo "ERROR: HALC evaluation failed; shards retained at $SHARD_ROOT" >&2
    exit 1
fi

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/merge_sharded_eval.py" \
    --final-output-dir "$FINAL_OUTPUT_DIR" \
    --shard-root "$SHARD_ROOT" \
    --eval-mode chair \
    --experiment-name "external-halc-${MODEL_FAMILY}-adapted-${MODEL_NAME}" \
    --seed 42
cp "${SHARD_ROOT}/shard_00/out/halc_config.json" "${FINAL_OUTPUT_DIR}/halc_config.json"

"$PYTHON_BIN" - "$FINAL_OUTPUT_DIR" "$TEST_JSON" "$HALC_MAX_SAMPLES" <<'PY'
import json
import sys
from pathlib import Path

output_dir, test_json, max_samples = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
expected = len(json.loads(test_json.read_text()))
if max_samples:
    expected = min(expected, max_samples)
rows = [json.loads(line) for line in (output_dir / "eval_results.jsonl").read_text().splitlines() if line]
if len(rows) != expected or len({row["image_id"] for row in rows}) != expected:
    raise RuntimeError(f"HALC merge incomplete: expected={expected}, rows={len(rows)}")
triggered = sum(row["halc_triggered_count"] for row in rows)
changed = sum(row["halc_changed_token_count"] for row in rows)
summary = {"rows": len(rows), "triggered": triggered, "changed_tokens": changed,
           "status": "adaptation_unvalidated_against_upstream"}
(output_dir / "halc_diagnostics.json").write_text(json.dumps(summary, indent=2))
print(f"HALC_MERGE=PASS rows={len(rows)} triggered={triggered} changed={changed}")
if not triggered or not changed:
    raise RuntimeError("HALC adaptation did not demonstrably change any output; do not report as HALC")
PY

echo "HALC adapted CHAIR evaluation complete: $FINAL_OUTPUT_DIR"

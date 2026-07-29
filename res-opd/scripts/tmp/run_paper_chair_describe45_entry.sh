#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
MANIFEST="${PAPER_CHAIR_DESCRIBE_MANIFEST:-${RES_OPD_ROOT}/eval/manifests/paper_chair_describe45_sources.txt}"

ENTRY_INDEX="${1:?Usage: $0 <entry_index:1-45> [vllm_base_port]}"
VLLM_BASE_PORT="${2:-${VLLM_BASE_PORT:-8600}}"

if ! [[ "$ENTRY_INDEX" =~ ^[0-9]+$ ]] || (( ENTRY_INDEX < 1 || ENTRY_INDEX > 45 )); then
    echo "Error: entry_index must be an integer in [1, 45], got ${ENTRY_INDEX}" >&2
    exit 1
fi
if ! [[ "$VLLM_BASE_PORT" =~ ^[0-9]+$ ]]; then
    echo "Error: vllm_base_port must be an integer, got ${VLLM_BASE_PORT}" >&2
    exit 1
fi
if [[ ! -f "$MANIFEST" ]]; then
    echo "Error: source manifest not found: ${MANIFEST}" >&2
    exit 1
fi

line="$(awk -F'|' -v index="$ENTRY_INDEX" '$1 == index {print; exit}' "$MANIFEST")"
if [[ -z "$line" ]]; then
    echo "Error: entry ${ENTRY_INDEX} not found in ${MANIFEST}" >&2
    exit 1
fi
IFS='|' read -r index label source_type primary oss_name step student_ratio <<< "$line"

common_env=(
    DATASET_VERSION=full
    RESULT_VERSION_TAG=instruct
    DEGRADATION_MODE=original
    STUDENT_RATIO="$student_ratio"
    CHAIR_PROMPT="Describe the image."
    EVAL_OUTPUT_SUFFIX=chair_describe_image_official
    CHAIR_MAX_NEW_TOKENS=8192
    GPU_LIST=0,1,2,3,4,5,6,7
    EVAL_SHARD_COUNT=8
    VLLM_BASE_PORT="$VLLM_BASE_PORT"
    VLLM_PORT_CLEANUP=True
    SHARDED_EVAL_FORCE_TP1=True
    SHARDED_EVAL_KEEP_SHARDS=False
)

echo "============================================================"
echo "Paper CHAIR Describe rerun ${index}/45"
echo "Label:         ${label}"
echo "Source type:   ${source_type}"
echo "Student ratio: ${student_ratio}"
echo "Base port:     ${VLLM_BASE_PORT}"
echo "============================================================"

cd "$VISION_OPD_ROOT"
if [[ "$source_type" == "local_base" ]]; then
    if [[ ! -d "$primary" || ! -f "${primary}/config.json" ]]; then
        echo "Error: local base model is incomplete: ${primary}" >&2
        exit 1
    fi
    env "${common_env[@]}" \
        MODEL_PROFILE=qwen3vl_instruct \
        CLEANUP_LOCAL_CKPT=False \
        bash "${RES_OPD_ROOT}/scripts/eval_after_merge_sharded_8gpu.sh" \
        "$primary" 0 latest chair
elif [[ "$source_type" == "oss_checkpoint" ]]; then
    if [[ -z "$oss_name" || -z "$step" ]]; then
        echo "Error: OSS entry ${index} is missing oss_name or step" >&2
        exit 1
    fi
    env "${common_env[@]}" \
        bash "${RES_OPD_ROOT}/scripts/eval_batch_from_oss.sh" \
        --experiment-names "$primary" \
        --oss-names "$oss_name" \
        --step "$step" \
        --degradation-mode original \
        --student-ratio "$student_ratio" \
        --eval-mode chair \
        --eval-backend sharded \
        --eval-shard-count 8 \
        --gpu-list 0,1,2,3,4,5,6,7 \
        --result-version-tag instruct \
        --chair-prompt "Describe the image." \
        --eval-output-suffix chair_describe_image_official \
        --chair-max-new-tokens 8192 \
        --sharded-eval-keep-shards False
else
    echo "Error: unsupported source type for entry ${index}: ${source_type}" >&2
    exit 1
fi

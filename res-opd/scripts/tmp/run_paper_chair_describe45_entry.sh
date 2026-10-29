#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
MANIFEST="${PAPER_CHAIR_DESCRIBE_MANIFEST:-${RES_OPD_ROOT}/eval/manifests/paper_chair_describe45_sources.txt}"

ENTRY_INDEX="${1:?Usage: $0 <entry_index:1-45> [vllm_base_port]}"
VLLM_BASE_PORT="${2:-${VLLM_BASE_PORT:-8600}}"
CHAIR_RERUN_PROMPT="${CHAIR_RERUN_PROMPT:-Describe the image.}"
CHAIR_RERUN_OUTPUT_SUFFIX="${CHAIR_RERUN_OUTPUT_SUFFIX:-chair_describe_image_official}"
MODEL_ROOT="${MODEL_ROOT:-}"
RERUN_GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
RERUN_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}"
OSS_CHECKPOINT_ACCESS_MODE="${OSS_CHECKPOINT_ACCESS_MODE:-mounted}"
OSS_MOUNT_ROOT="${OSS_MOUNT_ROOT:-/data/oss_bucket_0}"

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

line="$(awk -F'|' -v entry_id="$ENTRY_INDEX" '$1 == entry_id {print; exit}' "$MANIFEST")"
if [[ -z "$line" ]]; then
    echo "Error: entry ${ENTRY_INDEX} not found in ${MANIFEST}" >&2
    exit 1
fi
IFS='|' read -r entry_id label source_type primary oss_name step student_ratio <<< "$line"
if [[ "$source_type" == "local_base" && -n "$MODEL_ROOT" ]]; then
    primary="${MODEL_ROOT%/}/$(basename "$primary")"
fi

common_env=(
    DATASET_VERSION=full
    RESULT_VERSION_TAG=instruct
    MODEL_PROFILE=qwen3vl_instruct
    DEGRADATION_MODE=original
    STUDENT_RATIO="$student_ratio"
    CHAIR_PROMPT="$CHAIR_RERUN_PROMPT"
    EVAL_OUTPUT_SUFFIX="$CHAIR_RERUN_OUTPUT_SUFFIX"
    CHAIR_MAX_NEW_TOKENS=8192
    GPU_LIST="$RERUN_GPU_LIST"
    EVAL_SHARD_COUNT="$RERUN_SHARD_COUNT"
    VLLM_BASE_PORT="$VLLM_BASE_PORT"
    VLLM_PORT_CLEANUP=True
    SHARDED_EVAL_FORCE_TP1=True
    SHARDED_EVAL_KEEP_SHARDS=False
    OSS_CHECKPOINT_ACCESS_MODE="$OSS_CHECKPOINT_ACCESS_MODE"
    OSS_MOUNT_ROOT="$OSS_MOUNT_ROOT"
)

echo "============================================================"
echo "Paper CHAIR Describe rerun ${entry_id}/45"
echo "Label:         ${label}"
echo "Source type:   ${source_type}"
echo "Student ratio: ${student_ratio}"
echo "Base port:     ${VLLM_BASE_PORT}"
echo "Checkpoint:    ${OSS_CHECKPOINT_ACCESS_MODE} (${OSS_MOUNT_ROOT})"
echo "============================================================"

cd "$VISION_OPD_ROOT"
if [[ "$source_type" == "local_base" ]]; then
    if [[ ! -d "$primary" || ! -f "${primary}/config.json" ]]; then
        echo "Error: local base model is incomplete: ${primary}" >&2
        exit 1
    fi
    env "${common_env[@]}" \
        CLEANUP_LOCAL_CKPT=False \
        bash "${RES_OPD_ROOT}/scripts/eval_after_merge_sharded_8gpu.sh" \
        "$primary" 0 latest chair
elif [[ "$source_type" == "oss_checkpoint" ]]; then
    if [[ -z "$oss_name" || -z "$step" ]]; then
        echo "Error: OSS entry ${entry_id} is missing oss_name or step" >&2
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
        --eval-shard-count "$RERUN_SHARD_COUNT" \
        --gpu-list "$RERUN_GPU_LIST" \
        --result-version-tag instruct \
        --chair-prompt "$CHAIR_RERUN_PROMPT" \
        --eval-output-suffix "$CHAIR_RERUN_OUTPUT_SUFFIX" \
        --chair-max-new-tokens 8192 \
        --sharded-eval-keep-shards False
else
    echo "Error: unsupported source type for entry ${entry_id}: ${source_type}" >&2
    exit 1
fi

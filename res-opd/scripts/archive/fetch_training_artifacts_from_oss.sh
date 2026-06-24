#!/bin/bash

set -euo pipefail

# =============================================================================
# Fetch Res-OPD training artifacts from OSS.
#
# Examples:
#   bash res-opd/scripts/fetch_training_artifacts_from_oss.sh \
#       --experiment-name Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a0.5-ema-e1
#
#   bash res-opd/scripts/fetch_training_artifacts_from_oss.sh \
#       --experiment-name Res-OPD-... \
#       --step global_step_40
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
EXPERIMENT_NAME=""
OSS_NAME=""
STEP=""
OUTPUT_ROOT="$RES_OPD_ROOT"

usage() {
    cat >&2 <<EOF
Usage: $0 --experiment-name EXPERIMENT_NAME [--step global_step_N] [--oss-name OSS_NAME] [--output-root DIR]

Options:
  --experiment-name  Local experiment name, used for output dirs and OSS-name inference.
  --oss-name         Explicit OSS experiment name. If omitted, inferred from --experiment-name.
  --step             Optional checkpoint step to download, e.g. global_step_40.
  --output-root      Root containing traces/, rollouts/, logs/, checkpoints/. Default: res-opd root.

Environment:
  OSS_BASE           OSS root. Default: oss://industry-algo/yanlin/ckpt/OPD/v4
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --experiment-name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --oss-name)
            OSS_NAME="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

get_oss_name() {
    local ckpt_dir_name="$1"
    local suffix
    suffix="${ckpt_dir_name#Res-OPD-Qwen3VL-2B-Instruct-}"
    if [[ "$suffix" == "$ckpt_dir_name" ]]; then
        suffix="$ckpt_dir_name"
    fi

    local epoch_tag=""
    if [[ "$suffix" =~ ^(.+)-(e[0-9]+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        epoch_tag="-${BASH_REMATCH[2]}"
    fi

    echo "ResOPD_${suffix//-/_}${epoch_tag}"
}

if [[ -z "$EXPERIMENT_NAME" && -z "$OSS_NAME" ]]; then
    echo "Error: --experiment-name or --oss-name is required." >&2
    usage
    exit 1
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
    EXPERIMENT_NAME="$OSS_NAME"
fi
if [[ -z "$OSS_NAME" ]]; then
    OSS_NAME="$(get_oss_name "$EXPERIMENT_NAME")"
fi

if ! command -v ossutil >/dev/null 2>&1; then
    echo "Error: ossutil not found in PATH." >&2
    exit 1
fi

OSS_EXP_PATH="${OSS_BASE%/}/${OSS_NAME}"
ARTIFACT_PATH="${OSS_EXP_PATH}/training_artifacts"

TRACE_DIR="${OUTPUT_ROOT%/}/traces/${EXPERIMENT_NAME}"
ROLLOUT_DIR="${OUTPUT_ROOT%/}/rollouts/${EXPERIMENT_NAME}"
LOG_DIR="${OUTPUT_ROOT%/}/logs"
CKPT_DIR="${OUTPUT_ROOT%/}/checkpoints/${EXPERIMENT_NAME}"

echo "============================================================"
echo " Fetch Res-OPD Artifacts from OSS"
echo "============================================================"
echo "Experiment: $EXPERIMENT_NAME"
echo "OSS name:   $OSS_NAME"
echo "OSS path:   $OSS_EXP_PATH"
echo "Output:     $OUTPUT_ROOT"
echo "============================================================"

mkdir -p "$TRACE_DIR" "$ROLLOUT_DIR" "$LOG_DIR"

echo "[1/4] Fetching traces ..."
ossutil cp -r "${ARTIFACT_PATH}/traces/" "${TRACE_DIR}/" -f || true

echo "[2/4] Fetching rollouts ..."
ossutil cp -r "${ARTIFACT_PATH}/rollouts/" "${ROLLOUT_DIR}/" -f || true

echo "[3/4] Fetching logs and metadata ..."
ossutil cp -r "${ARTIFACT_PATH}/logs/" "${LOG_DIR}/" -f || true
ossutil cp -r "${ARTIFACT_PATH}/checkpoint_metadata/" "${CKPT_DIR}/checkpoint_metadata/" -f || true
ossutil cp "${ARTIFACT_PATH}/manifest.txt" "${CKPT_DIR}/post_train_artifacts_manifest.txt" -f || true

if [[ -n "$STEP" ]]; then
    echo "[4/4] Fetching checkpoint ${STEP} ..."
    mkdir -p "${CKPT_DIR}/${STEP}"
    ossutil cp -r "${OSS_EXP_PATH}/${STEP}/" "${CKPT_DIR}/${STEP}/" -f
else
    echo "[4/4] No --step specified; checkpoint download skipped."
fi

echo "Done."
echo "Trace dir:   $TRACE_DIR"
echo "Rollout dir: $ROLLOUT_DIR"
echo "Ckpt dir:    $CKPT_DIR"

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_SERIES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$LLAVA_SERIES_DIR"

MODEL_SIZE="${MODEL_SIZE:-${1:-7b}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"

case "$MODEL_SIZE" in
    7b|llava-v1.5-7b|llava-7b)
        MODEL_ID="llava-v1.5-7b"
        DEFAULT_MODEL_PATH="liuhaotian/llava-v1.5-7b"
        ;;
    13b|llava-v1.5-13b|llava-13b)
        MODEL_ID="llava-v1.5-13b"
        DEFAULT_MODEL_PATH="liuhaotian/llava-v1.5-13b"
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE. Use 7b or 13b." >&2
        exit 1
        ;;
esac

GENERATE_SCRIPT="${GENERATE_SCRIPT:-generate/generate_llava.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/${MODEL_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/answers.jsonl}"
MERGED_LOGITS_FILE="${MERGED_LOGITS_FILE:-${OUTPUT_DIR}/logits_merged.pt}"
CONVERT_NAME="${CONVERT_NAME:-${MODEL_ID}}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
MAX_WORKERS="${MAX_WORKERS:-64}"
TOP_R="${TOP_R:-30}"

echo "Working directory: $LLAVA_SERIES_DIR"
echo "Model ID: $MODEL_ID"
echo "Generation script: $GENERATE_SCRIPT"
echo "Output directory: $OUTPUT_DIR"
echo "Answers JSONL: $OUTPUT_FILE"
echo "Merged logits: $MERGED_LOGITS_FILE"
echo "DPO output prefix: $CONVERT_NAME"

if [[ "$RUN_GENERATE" == "1" ]]; then
    echo
    echo "==================== Step 1/2: Generate answers and logits ===================="
    MODEL_PATH="$MODEL_PATH" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    OUTPUT_FILE="$OUTPUT_FILE" \
    MERGED_LOGITS_FILE="$MERGED_LOGITS_FILE" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$GENERATE_SCRIPT" "$MODEL_SIZE"
fi

if [[ "$RUN_CONVERT" == "1" ]]; then
    echo
    echo "==================== Step 2/2: Convert logits to DPO data ===================="
    convert_args=(
        generate/convert_llava.py
        --name "$CONVERT_NAME"
        --model-path "$MODEL_PATH"
        --max-workers "$MAX_WORKERS"
        --input-jsonl "$OUTPUT_FILE"
        --logits-path "$MERGED_LOGITS_FILE"
        --top-r "$TOP_R"
    )

    if [[ -n "${ENTROPY_OUTPUT_JSONL:-}" ]]; then
        convert_args+=(--entropy-output-jsonl "$ENTROPY_OUTPUT_JSONL")
    fi

    if [[ -n "${DPO_OUTPUT_JSON:-}" ]]; then
        convert_args+=(--dpo-output-json "$DPO_OUTPUT_JSON")
    fi

    if [[ "${FILTER_EQUAL_ENTROPY:-1}" == "0" ]]; then
        convert_args+=(--no-filter-equal-entropy)
    fi

    "$PYTHON_BIN" "${convert_args[@]}"
fi

echo
echo "Pipeline finished."

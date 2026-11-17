#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_SERIES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$HF_SERIES_DIR"

MODEL_SPEC="${MODEL_SPEC:-${1:-qwen25-3b}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_GENERATE="${RUN_GENERATE:-1}"
RUN_CONVERT="${RUN_CONVERT:-1}"

case "$MODEL_SPEC" in
    qwen25-3b|qwen2.5-vl-3b|qwen-2.5-vl-3b|3b)
        MODEL_ID="qwen2.5-vl-3b"
        DEFAULT_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
        CONVERT_SCRIPT="generate/convert_qwen.py"
        DEFAULT_TOP_R=20
        ;;
    qwen3-8b|qwen3-vl-8b|qwen3)
        MODEL_ID="qwen3-vl-8b"
        DEFAULT_MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
        CONVERT_SCRIPT="generate/convert_qwen.py"
        DEFAULT_TOP_R=5
        ;;
    internvl35-8b|internvl3_5-8b|internvl3.5-8b|intern)
        MODEL_ID="internvl3_5-8b"
        DEFAULT_MODEL_PATH="OpenGVLab/InternVL3_5-8B-HF"
        CONVERT_SCRIPT="generate/convert_intern.py"
        DEFAULT_TOP_R=40
        ;;
    *)
        echo "Unknown MODEL_SPEC: $MODEL_SPEC. Use qwen25-3b, qwen3-8b, or internvl35-8b." >&2
        exit 1
        ;;
esac

GENERATE_SCRIPT="${GENERATE_SCRIPT:-generate/generate_hf.sh}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/${MODEL_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/answers.jsonl}"
MERGED_LOGITS_FILE="${MERGED_LOGITS_FILE:-${OUTPUT_DIR}/logits_merged.pt}"
CONVERT_NAME="${CONVERT_NAME:-${MODEL_ID}}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
TOP_R="${TOP_R:-$DEFAULT_TOP_R}"
MAX_WORKERS="${MAX_WORKERS:-64}"
ENTROPY_OUTPUT_JSONL="${ENTROPY_OUTPUT_JSONL:-${OUTPUT_DIR}/${CONVERT_NAME}.jsonl}"
DPO_OUTPUT_JSON="${DPO_OUTPUT_JSON:-${OUTPUT_DIR}/${CONVERT_NAME}_dpo_data.json}"

echo "Working directory: $HF_SERIES_DIR"
echo "Model ID: $MODEL_ID"
echo "Generation script: $GENERATE_SCRIPT"
echo "Output directory: $OUTPUT_DIR"
echo "Answers JSONL: $OUTPUT_FILE"
echo "Merged logits: $MERGED_LOGITS_FILE"
echo "Convert script: $CONVERT_SCRIPT"
echo "DPO output: $DPO_OUTPUT_JSON"

if [[ "$RUN_GENERATE" == "1" ]]; then
    echo
    echo "==================== Step 1/2: Generate answers and logits ===================="
    MODEL_PATH="$MODEL_PATH" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    OUTPUT_FILE="$OUTPUT_FILE" \
    MERGED_LOGITS_FILE="$MERGED_LOGITS_FILE" \
    PYTHON_BIN="$PYTHON_BIN" \
    bash "$GENERATE_SCRIPT" "$MODEL_SPEC"
fi

if [[ "$RUN_CONVERT" == "1" ]]; then
    echo
    echo "==================== Step 2/2: Convert logits to DPO data ===================="
    convert_args=(
        "$CONVERT_SCRIPT"
        --name "$CONVERT_NAME"
        --model-path "$MODEL_PATH"
        --max-workers "$MAX_WORKERS"
        --input-jsonl "$OUTPUT_FILE"
        --logits-path "$MERGED_LOGITS_FILE"
        --top-r "$TOP_R"
        --entropy-output-jsonl "$ENTROPY_OUTPUT_JSONL"
        --dpo-output-json "$DPO_OUTPUT_JSON"
    )

    if [[ "${FILTER_EQUAL_ENTROPY:-1}" == "0" ]]; then
        convert_args+=(--no-filter-equal-entropy)
    fi

    "$PYTHON_BIN" "${convert_args[@]}"
fi

echo
echo "Pipeline finished."

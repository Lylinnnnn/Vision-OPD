#!/bin/bash
set -euo pipefail

cleanup() {
    echo ""
    echo "!!! Ctrl+C or termination signal detected !!!"
    echo "!!! Terminating child processes..."
    trap - SIGINT SIGTERM
    kill 0 2>/dev/null || true
    exit 1
}
trap cleanup SIGINT SIGTERM

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_SERIES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_DIR="$(cd "${LLAVA_SERIES_DIR}/.." && pwd)"
cd "$LLAVA_SERIES_DIR"

MODEL_SIZE="${MODEL_SIZE:-${1:-7b}}"
case "$MODEL_SIZE" in
    7b|llava-v1.5-7b|llava-7b)
        MODEL_ID="llava-v1.5-7b"
        DEFAULT_MODEL_PATH="liuhaotian/llava-v1.5-7b"
        DEFAULT_BATCH_SIZE=10
        ;;
    13b|llava-v1.5-13b|llava-13b)
        MODEL_ID="llava-v1.5-13b"
        DEFAULT_MODEL_PATH="liuhaotian/llava-v1.5-13b"
        DEFAULT_BATCH_SIZE=5
        ;;
    *)
        echo "Unknown MODEL_SIZE: $MODEL_SIZE. Use 7b or 13b." >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${TRAIN_DIR}/RLAIF_dataset}"
QUESTION_FILE="${QUESTION_FILE:-${TRAIN_DIR}/RLAIF_dataset/rlaif10k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/${MODEL_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/answers.jsonl}"
MERGED_LOGITS_FILE="${MERGED_LOGITS_FILE:-${OUTPUT_DIR}/logits_merged.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
NUM_SAMPLES="${NUM_SAMPLES:-10}"
TOP_K_LOGITS="${TOP_K_LOGITS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INFER_SCRIPT="${INFER_SCRIPT:-generate/llava_infer.py}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "$QUESTION_FILE" ]]; then
    echo "Question file not found: $QUESTION_FILE" >&2
    exit 1
fi

if [[ "$MODEL_PATH" == /* && ! -e "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "LLaVA inference script not found: $INFER_SCRIPT" >&2
    exit 1
fi

GPU_IDS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' ' ')
GPU_ARRAY=($GPU_IDS)
NUM_WORKERS=${#GPU_ARRAY[@]}
NUM_CHUNKS="${NUM_CHUNKS:-$NUM_WORKERS}"

if [[ "$NUM_WORKERS" -eq 0 ]]; then
    echo "CUDA_VISIBLE_DEVICES is empty; cannot start inference." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Working directory: $LLAVA_SERIES_DIR"
echo "Model ID: $MODEL_ID"
echo "Model path: $MODEL_PATH"
echo "GPU list: $CUDA_VISIBLE_DEVICES"
echo "Workers: $NUM_WORKERS, chunks: $NUM_CHUNKS"
echo "batch size: $BATCH_SIZE"
echo "Question file: $QUESTION_FILE"
echo "Output JSONL: $OUTPUT_FILE"
echo "Output logits: $MERGED_LOGITS_FILE"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1; showing configuration only, not starting inference."
    exit 0
fi

rm -f "${OUTPUT_DIR}"/temp_*.jsonl
rm -f "${OUTPUT_DIR}"/all_logits_chunk*_merged.pt

pids=()
for ((i=0; i<NUM_WORKERS; i++)); do
    GPU_ID=${GPU_ARRAY[i]}
    CHUNK_IDX=$i
    TEMP_ANS_FILE="${OUTPUT_DIR}/temp_${CHUNK_IDX}.jsonl"

    CUDA_VISIBLE_DEVICES=$GPU_ID "$PYTHON_BIN" "$INFER_SCRIPT" \
        --model-name "$MODEL_PATH" \
        --image-folder "$IMAGE_FOLDER" \
        --question-file "$QUESTION_FILE" \
        --answers-file "$TEMP_ANS_FILE" \
        --num-chunks "$NUM_CHUNKS" \
        --chunk-idx "$CHUNK_IDX" \
        --temperature "$TEMPERATURE" \
        --num-samples "$NUM_SAMPLES" \
        --top-k-logits "$TOP_K_LOGITS" \
        --conv-mode "$CONV_MODE" \
        --batch-size "$BATCH_SIZE" &

    pids+=($!)
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done

if [[ "$failed" -ne 0 ]]; then
    echo "At least one inference process failed; stopping merge." >&2
    exit 1
fi

shopt -s nullglob
TEMP_FILES=("${OUTPUT_DIR}"/temp_*.jsonl)
if [[ "${#TEMP_FILES[@]}" -eq 0 ]]; then
    echo "No temporary JSONL files found; stopping merge." >&2
    exit 1
fi
cat "${TEMP_FILES[@]}" > "$OUTPUT_FILE"
rm -f "${OUTPUT_DIR}"/temp_*.jsonl

echo "Merging logits files..."
"$PYTHON_BIN" - <<PY
import torch
from pathlib import Path

output_dir = Path('${OUTPUT_DIR}')
merged = {}
files = sorted(output_dir.glob('all_logits_chunk*_merged.pt'))
if not files:
    raise SystemExit('No logits chunk files found; stopping merge.')

for f in files:
    print(f'  Loading: {f.name}')
    try:
        chunk = torch.load(f, map_location='cpu', weights_only=False)
    except TypeError:
        chunk = torch.load(f, map_location='cpu')
    merged.update(chunk)

torch.save(merged, '${MERGED_LOGITS_FILE}')
print(f'Merge complete: {len(merged)} records')
for f in files:
    f.unlink()
print(f'Cleaned {len(files)} temporary logits chunk files')
PY

echo "Done."
echo "JSONL results saved to: $OUTPUT_FILE"
echo "Logits data saved to: $MERGED_LOGITS_FILE"

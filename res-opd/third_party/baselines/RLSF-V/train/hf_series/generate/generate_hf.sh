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
HF_SERIES_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_DIR="$(cd "${HF_SERIES_DIR}/.." && pwd)"
cd "$HF_SERIES_DIR"

MODEL_SPEC="${MODEL_SPEC:-${1:-qwen25-3b}}"
case "$MODEL_SPEC" in
    qwen25-3b|qwen2.5-vl-3b|qwen-2.5-vl-3b|3b)
        MODEL_ID="qwen2.5-vl-3b"
        DEFAULT_MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
        DEFAULT_QUESTION_FILE="${TRAIN_DIR}/RLAIF_dataset/rlaif10k.jsonl"
        DEFAULT_BATCH_SIZE=10
        DEFAULT_NUM_SAMPLES=10
        INFER_SCRIPT="${INFER_SCRIPT:-generate/qwen25_infer.py}"
        ;;
    qwen3-8b|qwen3-vl-8b|qwen3)
        MODEL_ID="qwen3-vl-8b"
        DEFAULT_MODEL_PATH="Qwen/Qwen3-VL-8B-Instruct"
        DEFAULT_QUESTION_FILE="${TRAIN_DIR}/RLAIF_dataset/rlaif10k.jsonl"
        DEFAULT_BATCH_SIZE=10
        DEFAULT_NUM_SAMPLES=10
        INFER_SCRIPT="${INFER_SCRIPT:-generate/qwen3_infer.py}"
        ;;
    internvl35-8b|internvl3_5-8b|internvl3.5-8b|intern)
        MODEL_ID="internvl3_5-8b"
        DEFAULT_MODEL_PATH="OpenGVLab/InternVL3_5-8B-HF"
        DEFAULT_QUESTION_FILE="${TRAIN_DIR}/RLAIF_dataset/rlaif10k.jsonl"
        DEFAULT_BATCH_SIZE=16
        DEFAULT_NUM_SAMPLES=10
        INFER_SCRIPT="${INFER_SCRIPT:-generate/internvl3_5_infer.py}"
        ;;
    *)
        echo "Unknown MODEL_SPEC: $MODEL_SPEC. Use qwen25-3b, qwen3-8b, or internvl35-8b." >&2
        exit 1
        ;;
esac

MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${TRAIN_DIR}/RLAIF_dataset}"
QUESTION_FILE="${QUESTION_FILE:-$DEFAULT_QUESTION_FILE}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/${MODEL_ID}}"
OUTPUT_FILE="${OUTPUT_FILE:-${OUTPUT_DIR}/answers.jsonl}"
MERGED_LOGITS_FILE="${MERGED_LOGITS_FILE:-${OUTPUT_DIR}/logits_merged.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
BATCH_SIZE="${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}"
NUM_SAMPLES="${NUM_SAMPLES:-$DEFAULT_NUM_SAMPLES}"
TOP_K_LOGITS="${TOP_K_LOGITS:-50}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

WAIT_FOR_GPUS="${WAIT_FOR_GPUS:-0}"
WAIT_GPU_IDS="${WAIT_GPU_IDS:-$CUDA_VISIBLE_DEVICES}"
MEM_THRESHOLD="${MEM_THRESHOLD:-1000}"
CHECK_INTERVAL="${CHECK_INTERVAL:-30}"

if [[ "$WAIT_FOR_GPUS" == "1" ]]; then
    IFS=',' read -ra TARGET_GPUS <<< "$WAIT_GPU_IDS"

    echo "=========================================="
    echo " Waiting for GPU ${TARGET_GPUS[*]} to become idle..."
    echo " Memory threshold: ${MEM_THRESHOLD} MiB"
    echo " Check interval: ${CHECK_INTERVAL} seconds"
    echo "=========================================="

    while true; do
        all_free=true
        for gpu_id in "${TARGET_GPUS[@]}"; do
            used_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
            if [[ -z "$used_mem" ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: cannot read GPU $gpu_id status"
                all_free=false
                break
            fi
            if (( used_mem > MEM_THRESHOLD )); then
                all_free=false
                break
            fi
        done

        if $all_free; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Target GPUs are idle; starting inference."
            break
        fi

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] GPU still busy; checking again in ${CHECK_INTERVAL}s..."
        sleep "$CHECK_INTERVAL"
    done
fi

if [[ ! -f "$QUESTION_FILE" ]]; then
    echo "Question file not found: $QUESTION_FILE" >&2
    exit 1
fi

if [[ "$MODEL_PATH" == /* && ! -e "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ ! -f "$INFER_SCRIPT" ]]; then
    echo "HF inference script not found: $INFER_SCRIPT" >&2
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

echo "Working directory: $HF_SERIES_DIR"
echo "Model ID: $MODEL_ID"
echo "Model path: $MODEL_PATH"
echo "Inference script: $INFER_SCRIPT"
echo "GPU list: $CUDA_VISIBLE_DEVICES"
echo "Workers: $NUM_WORKERS, chunks: $NUM_CHUNKS"
echo "batch size: $BATCH_SIZE"
echo "num samples: $NUM_SAMPLES"
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
        --max-new-tokens "$MAX_NEW_TOKENS" \
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

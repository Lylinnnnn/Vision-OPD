#!/bin/bash
# =============================================================================
# Batch eval: Launch 6 vLLM servers + CHAIR eval in parallel (1 GPU each)
# Usage: bash scripts/run_eval_batch.sh
# =============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

CKPT_DIR="${RES_OPD_ROOT}/eval_ckpts"
OUTPUT_BASE="${RES_OPD_ROOT}/eval_results_v4"
TEST_JSON="${RES_OPD_ROOT}/data/test.json"
EVAL_SCRIPT="${RES_OPD_ROOT}/eval/eval_chair.py"
PYTHON="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3"

mkdir -p "$OUTPUT_BASE"

# Define experiments: name|ckpt_subdir|student_px|gpu_id|port
EXPERIMENTS=(
    "s224_rkl_step120|s224_rkl_step120|224|0|8000"
    "s448_t224_rkl_step40|s448_t224_rkl_step40|448|1|8001"
    "s336_rkl_step80|s336_rkl_step80|336|2|8002"
    "s448_t336_rkl_step140|s448_t336_rkl_step140|448|3|8003"
    "s224_jsd_step160|s224_jsd_step160|224|4|8004"
    "s336_jsd_step100|s336_jsd_step100|336|5|8005"
)

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME CKPT_SUBDIR STUDENT_PX GPU_ID PORT <<< "$exp"
    
    CKPT_PATH="${CKPT_DIR}/${CKPT_SUBDIR}"
    OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    TMUX_NAME="eval-${NAME}"
    
    if [ ! -f "${CKPT_PATH}/model.safetensors" ]; then
        echo "ERROR: Checkpoint not found: ${CKPT_PATH}/model.safetensors"
        continue
    fi
    
    mkdir -p "$OUTPUT_DIR"
    
    # Kill existing tmux session if any
    tmux kill-session -t "$TMUX_NAME" 2>/dev/null || true
    
    # Create tmux session with vLLM + eval pipeline
    tmux new-session -d -s "$TMUX_NAME" bash -c "
        set -e
        export CUDA_VISIBLE_DEVICES=${GPU_ID}
        
        echo '[${NAME}] Starting vLLM on GPU ${GPU_ID}, port ${PORT}...'
        ${PYTHON} -m vllm.entrypoints.openai.api_server \
            --model '${CKPT_PATH}' \
            --gpu-memory-utilization 0.85 \
            --served-model-name 'Res-OPD' \
            --trust-remote-code \
            --port ${PORT} \
            --max-model-len 9728 &
        VLLM_PID=\$!
        
        # Wait for server ready
        for i in \$(seq 1 300); do
            if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
                echo '[${NAME}] vLLM ready after \${i}s'
                break
            fi
            if ! kill -0 \$VLLM_PID 2>/dev/null; then
                echo '[${NAME}] vLLM exited unexpectedly!'
                exit 1
            fi
            sleep 1
        done
        
        if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
            echo '[${NAME}] vLLM failed to start'
            kill \$VLLM_PID 2>/dev/null || true
            exit 1
        fi
        
        echo '[${NAME}] Running CHAIR eval (student_px=${STUDENT_PX})...'
        ${PYTHON} '${EVAL_SCRIPT}' \
            --api-base 'http://localhost:${PORT}/v1/' \
            --model-name 'Res-OPD' \
            --test-json '${TEST_JSON}' \
            --output-dir '${OUTPUT_DIR}' \
            --student-px ${STUDENT_PX} \
            --target-px 448 \
            --parallel-workers 8
        
        echo '[${NAME}] Eval complete!'
        kill \$VLLM_PID 2>/dev/null || true
        wait \$VLLM_PID 2>/dev/null || true
        echo '[${NAME}] Done.'
    "
    
    echo "Started tmux session: $TMUX_NAME (GPU=$GPU_ID, port=$PORT, student_px=$STUDENT_PX)"
done

echo ""
echo "All 6 eval sessions launched. Monitor with:"
echo "  tmux ls"
echo "  tmux attach -t eval-s224_rkl_step120"
echo ""
echo "Results will be saved to: $OUTPUT_BASE/"

#!/bin/bash
# Launch 6 parallel CHAIR eval sessions (1 GPU each)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3"
CKPT_DIR="${RES_OPD_ROOT}/eval_ckpts"
OUTPUT_BASE="${RES_OPD_ROOT}/eval_results_v4"
TEST_JSON="${RES_OPD_ROOT}/data/test.json"
EVAL_SCRIPT="${RES_OPD_ROOT}/eval/eval_chair.py"

mkdir -p "$OUTPUT_BASE"

# Kill old sessions
for s in eval-s224_rkl eval-s448t224_rkl eval-s336_rkl eval-s448t336_rkl eval-s224_jsd eval-s336_jsd; do
    tmux kill-session -t "$s" 2>/dev/null || true
done
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
sleep 2

launch_eval() {
    local NAME="$1" CKPT_SUBDIR="$2" STUDENT_PX="$3" GPU_ID="$4" PORT="$5"
    local CKPT_PATH="${CKPT_DIR}/${CKPT_SUBDIR}"
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    local TMUX_NAME="eval-${NAME}"
    local LOG_FILE="/tmp/eval_${NAME}.log"

    mkdir -p "$OUTPUT_DIR"

    # Write a self-contained script for this experiment
    local JOB_SCRIPT="/tmp/eval_job_${NAME}.sh"
    cat > "$JOB_SCRIPT" << JOBEOF
#!/bin/bash
export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PATH="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin:\${PATH}"

echo "[${NAME}] GPU=${GPU_ID} port=${PORT} student_px=${STUDENT_PX}"
echo "[${NAME}] Starting vLLM ..."

${PYTHON} -u -m vllm.entrypoints.openai.api_server \
    --model "${CKPT_PATH}" \
    --gpu-memory-utilization 0.85 \
    --served-model-name Res-OPD \
    --trust-remote-code \
    --port ${PORT} \
    --max-model-len 9728 &
VLLM_PID=\$!

# Wait for server ready (up to 5 min)
for i in \$(seq 1 300); do
    if curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
        echo "[${NAME}] vLLM ready after \${i}s"
        break
    fi
    if ! kill -0 \$VLLM_PID 2>/dev/null; then
        echo "[${NAME}] ERROR: vLLM exited!"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:${PORT}/health > /dev/null 2>&1; then
    echo "[${NAME}] ERROR: vLLM failed to start within 300s"
    kill \$VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "[${NAME}] Running CHAIR eval ..."
${PYTHON} -u "${EVAL_SCRIPT}" \
    --api-base "http://localhost:${PORT}/v1/" \
    --model-name Res-OPD \
    --test-json "${TEST_JSON}" \
    --output-dir "${OUTPUT_DIR}" \
    --student-px ${STUDENT_PX} \
    --target-px 448 \
    --parallel-workers 8

echo "[${NAME}] Eval complete! Shutting down vLLM ..."
kill \$VLLM_PID 2>/dev/null || true
wait \$VLLM_PID 2>/dev/null || true
echo "[${NAME}] DONE."
JOBEOF
    chmod +x "$JOB_SCRIPT"

    tmux new-session -d -s "$TMUX_NAME" "bash $JOB_SCRIPT 2>&1 | tee $LOG_FILE"
    echo "  Launched: $TMUX_NAME (GPU=$GPU_ID, port=$PORT, student_px=$STUDENT_PX)"
}

echo "Launching 6 parallel eval sessions ..."
launch_eval "s224_rkl"       "s224_rkl_step120"       224 0 8000
launch_eval "s448t224_rkl"   "s448_t224_rkl_step40"   448 1 8001
launch_eval "s336_rkl"       "s336_rkl_step80"        336 2 8002
launch_eval "s448t336_rkl"   "s448_t336_rkl_step140"  448 3 8003
launch_eval "s224_jsd"       "s224_jsd_step160"       224 4 8004
launch_eval "s336_jsd"       "s336_jsd_step100"       336 5 8005

echo ""
echo "All 6 sessions launched!"
echo "Monitor: tmux ls"
echo "Attach:  tmux attach -t eval-s224_rkl"
echo "Logs:    /tmp/eval_*.log"
echo "Results: $OUTPUT_BASE/"

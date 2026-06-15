#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=5
echo "[s336_jsd_step100] Starting vLLM on GPU 5, port 8005..."
/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model '/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_ckpts/s336_jsd_step100' \
    --gpu-memory-utilization 0.85 \
    --served-model-name 'Res-OPD' \
    --trust-remote-code \
    --port 8005 \
    --max-model-len 9728 &
VLLM_PID=$!

for i in $(seq 1 300); do
    if curl -s http://localhost:8005/health > /dev/null 2>&1; then
        echo "[s336_jsd_step100] vLLM ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[s336_jsd_step100] vLLM exited unexpectedly!"
        exit 1
    fi
    sleep 1
done

if ! curl -s http://localhost:8005/health > /dev/null 2>&1; then
    echo "[s336_jsd_step100] vLLM failed to start"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

echo "[s336_jsd_step100] Running CHAIR eval (student_px=336)..."
/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 '/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval/eval_chair.py' \
    --api-base 'http://localhost:8005/v1/' \
    --model-name 'Res-OPD' \
    --test-json '/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/data/test.json' \
    --output-dir '/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results_v4/s336_jsd_step100' \
    --student-px 336 \
    --target-px 448 \
    --parallel-workers 8

echo "[s336_jsd_step100] Eval complete!"
kill $VLLM_PID 2>/dev/null || true
wait $VLLM_PID 2>/dev/null || true
echo "[s336_jsd_step100] Done."

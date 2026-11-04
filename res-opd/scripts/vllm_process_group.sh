#!/usr/bin/env bash

res_opd_stop_vllm_group() {
    local pid="$1"
    local wait_seconds="${2:-30}"
    local remaining

    echo "[cleanup] Shutting down vLLM process group ${pid} ..."
    if kill -0 -- "-$pid" 2>/dev/null; then
        kill -TERM -- "-$pid" 2>/dev/null || true
    elif kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
    fi

    for ((remaining=0; remaining<wait_seconds; remaining++)); do
        if ! kill -0 -- "-$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done

    echo "[cleanup] vLLM group ${pid} did not exit; sending SIGKILL ..." >&2
    kill -KILL -- "-$pid" 2>/dev/null || true
    for ((remaining=0; remaining<5; remaining++)); do
        if ! kill -0 -- "-$pid" 2>/dev/null; then
            return 0
        fi
        sleep 1
    done
    echo "Error: vLLM process group ${pid} still exists after cleanup." >&2
    return 1
}

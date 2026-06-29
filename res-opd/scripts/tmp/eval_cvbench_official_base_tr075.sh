#!/usr/bin/env bash
set -euo pipefail

# Clean CV-Bench rerun for base + tr0.75 variants.
# Results are written under each checkpoint's normal eval_results/latest folder:
#   <ckpt_result_dir>/cv-bench/{model_answer,judge}/...

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export VISION_BENCHMARK="${VISION_BENCHMARK:-cv-bench}"
export VERSION_TAG="${VERSION_TAG:-latest}"
export VISION_BENCHMARK_REFRESH_PROMPTS="${VISION_BENCHMARK_REFRESH_PROMPTS:-True}"
export VISION_BENCHMARK_OUTPUT_SUFFIX="${VISION_BENCHMARK_OUTPUT_SUFFIX:-}"
export RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-True}"
export MCQ_EXTRACT_MODE="${MCQ_EXTRACT_MODE:-official}"
export VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-16}"
export VISION_PARALLEL_WORKERS="${VISION_PARALLEL_WORKERS:-128}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-8040}"
export VLLM_PORT_CLEANUP="${VLLM_PORT_CLEANUP:-True}"
export VLLM_PORT_CLEANUP_WAIT="${VLLM_PORT_CLEANUP_WAIT:-20}"
export FORCE_VISION_EVAL="${FORCE_VISION_EVAL:-True}"

is_truthy() {
    [[ "$1" == "True" || "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "Y" || "$1" == "y" ]]
}

port_listener_pids() {
    local port="$1"
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
    elif command -v ss >/dev/null 2>&1; then
        pids="$(ss -ltnp "sport = :${port}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' || true)"
    fi
    for pid in $pids; do
        [[ -n "$pid" ]] && echo "$pid"
    done | sort -u
}

pid_cmdline() {
    local pid="$1"
    if [[ -r "/proc/${pid}/cmdline" ]]; then
        tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true
    else
        ps -p "$pid" -o args= 2>/dev/null || true
    fi
}

is_vllm_cmd() {
    local cmd="$1"
    [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* || "$cmd" == *" vllm "* || "$cmd" == *"/vllm"* ]]
}

check_or_cleanup_vllm_ports() {
    local phase="$1"
    local ports=("$VLLM_BASE_PORT" "$((VLLM_BASE_PORT + 1))" "$((VLLM_BASE_PORT + 2))" "$((VLLM_BASE_PORT + 3))")
    local port pid pids cmd blocked

    echo ""
    echo "[vLLM ${phase}] Checking ports: ${ports[*]}"
    for port in "${ports[@]}"; do
        pids="$(port_listener_pids "$port")"
        [[ -z "$pids" ]] && {
            echo "  port ${port}: free"
            continue
        }

        echo "  port ${port}: busy by pid(s): ${pids}"
        blocked=0
        for pid in $pids; do
            cmd="$(pid_cmdline "$pid")"
            if is_vllm_cmd "$cmd"; then
                if is_truthy "$VLLM_PORT_CLEANUP"; then
                    echo "    killing stale vLLM pid=${pid}: ${cmd}"
                    kill "$pid" 2>/dev/null || true
                else
                    echo "Error: port ${port} has vLLM pid=${pid}, but VLLM_PORT_CLEANUP=${VLLM_PORT_CLEANUP}" >&2
                    blocked=1
                fi
            else
                echo "Error: port ${port} is used by a non-vLLM process pid=${pid}: ${cmd}" >&2
                blocked=1
            fi
        done
        [[ "$blocked" -ne 0 ]] && exit 1

        for _ in $(seq 1 "$VLLM_PORT_CLEANUP_WAIT"); do
            pids="$(port_listener_pids "$port")"
            [[ -z "$pids" ]] && break
            sleep 1
        done

        pids="$(port_listener_pids "$port")"
        if [[ -n "$pids" ]]; then
            echo "    force-killing remaining stale vLLM pid(s) on port ${port}: ${pids}"
            for pid in $pids; do
                cmd="$(pid_cmdline "$pid")"
                is_vllm_cmd "$cmd" && kill -9 "$pid" 2>/dev/null || true
            done
        fi
    done
}

post_run_check() {
    check_or_cleanup_vllm_ports "post-run"
}

trap post_run_check EXIT
check_or_cleanup_vllm_ports "preflight"

bash "$SCRIPT_DIR/eval_mmstar_cvbench_base_tr075.sh"

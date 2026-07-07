#!/usr/bin/env bash

set -euo pipefail

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ⚠️  IMPORTANT: 一次性/临时 eval 脚本（如批量eval、对比实验等）必须放在
#     scripts/tmp/ 目录下，禁止直接放在 scripts/ 根目录！
#     scripts/ 只保留常规最小运行脚本。用完的临时脚本请及时删除或归档。
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Evaluate a merged Res-OPD checkpoint using CHAIR, POPE, and optional Vision-OPD benchmarks
#
# Steps:
#   1. Start vLLM server with the merged checkpoint
#   2. Run selected evaluation(s) via API
#   3. Shut down vLLM server
#
# Usage:
#   bash scripts/eval_after_merge.sh <merged_checkpoint_path> [student_px] [version_tag] [eval_mode]
#
# eval_mode:
#   chair              Run CHAIR only (default; preserves the original behavior)
#   pope               Run POPE only
#   frequent           Run CHAIR + POPE
#   vision             Run Vision-OPD eval/run_eval.sh benchmarks
#   chair,pope,vision  Run selected tasks
#   all                Run CHAIR + POPE + Vision-OPD benchmarks
#
# Output directory structure (unified naming):
#   res-opd/eval_results/<version_tag>/<experiment_name>_<step_tag>/<dataset_tag>/
#
#   - version_tag:  e.g. "v5", "v4_frozen", "debug" (default: "latest")
#   - experiment_name: derived from checkpoint parent dir name
#   - step_tag: e.g. "global_step_46" (auto-detected from path)
#   - dataset_tag: auto-derived from data files, e.g. "train1500_test300"
#
# Examples:
#   # Default version tag "latest"
#   bash scripts/eval_after_merge.sh \
#       ./checkpoints/Res-OPD-Qwen3VL-2B-Instruct-s448-t200-a0.5-ema-e2/global_step_92/ \
#       448
#   # → eval_results/latest/Res-OPD-...-s448-t200-..._global_step_92/train1500_test300/
#
#   # Custom version tag
#   bash scripts/eval_after_merge.sh \
#       ./checkpoints/Res-OPD-.../global_step_92/ 448 v5
#   # → eval_results/v5/Res-OPD-..._global_step_92/train1500_test300/
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [student_px] [version_tag]}"
STUDENT_PX="${2:-${STUDENT_PX:-}}"
VERSION_TAG="${3:-latest}"
RESULT_VERSION_TAG="${RESULT_VERSION_TAG:-}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-1}"
EVAL_SHARD_INDEX="${EVAL_SHARD_INDEX:-0}"
EVAL_BACKEND="${EVAL_BACKEND:-single}"
EVAL_SHARDED="${EVAL_SHARDED:-False}"
# Support both legacy (test.json) and full-scale (test_1000.json) datasets.
# Set DATASET_VERSION=full to use the full-scale dataset; default is full.
DATASET_VERSION="${DATASET_VERSION:-full}"
EVAL_MODE="${4:-${EVAL_MODE:-chair}}"
if [[ "${EVAL_SHARDED_CHILD:-0}" != "1" ]]; then
    if [[ "$EVAL_BACKEND" == "sharded" || "$EVAL_SHARDED" == "True" || "$EVAL_SHARDED" == "true" || "$EVAL_SHARDED" == "1" ]]; then
        exec "${RES_OPD_ROOT}/scripts/eval_after_merge_sharded_8gpu.sh" "$MODEL_PATH" "$STUDENT_PX" "$VERSION_TAG" "$EVAL_MODE"
    fi
fi
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
CLEANUP_LOCAL_CKPT_WAS_SET="${CLEANUP_LOCAL_CKPT+x}"
CLEANUP_LOCAL_CKPT="${CLEANUP_LOCAL_CKPT:-True}"
ALLOW_CLEANUP_OUTSIDE_CKPT_BASE="${ALLOW_CLEANUP_OUTSIDE_CKPT_BASE:-False}"
DEGRADATION_MODE_WAS_SET="${DEGRADATION_MODE+x}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO_WAS_SET="${STUDENT_RATIO+x}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
TARGET_PX_WAS_SET="${TARGET_PX+x}"
TARGET_PX="${TARGET_PX:-448}"
PORT="${VLLM_PORT:-8000}"
VLLM_PORT_CLEANUP="${VLLM_PORT_CLEANUP:-True}"
VLLM_PORT_CLEANUP_WAIT="${VLLM_PORT_CLEANUP_WAIT:-20}"
VLLM_GPU_MEMORY_UTILIZATION_WAS_SET="${VLLM_GPU_MEMORY_UTILIZATION+x}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN_WAS_SET="${VLLM_MAX_MODEL_LEN+x}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-9728}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
VLLM_MAX_NUM_SEQS_WAS_SET="${VLLM_MAX_NUM_SEQS+x}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}"
VLLM_MAX_NUM_BATCHED_TOKENS_WAS_SET="${VLLM_MAX_NUM_BATCHED_TOKENS+x}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
VLLM_CUSTOM_OPS_WAS_SET="${VLLM_CUSTOM_OPS+x}"
VLLM_CUSTOM_OPS="${VLLM_CUSTOM_OPS:-}"
VLLM_TENSOR_PARALLEL_SIZE_WAS_SET="${VLLM_TENSOR_PARALLEL_SIZE+x}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-}"
MODEL_PROFILE="${MODEL_PROFILE:-}"
MODEL_NAME="Res-OPD"
if [[ "$DATASET_VERSION" == "full" ]]; then
    TEST_JSON="${RES_OPD_ROOT}/data/test_1000.json"
else
    TEST_JSON="${RES_OPD_ROOT}/data/test.json"
fi
PYTHON_BIN="/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3"
POPE_BENCHMARK="${POPE_BENCHMARK:-pope_adv,pope_pop,pope_random}"
POPE_SOURCE="${POPE_SOURCE:-res-opd-test}"
POPE_QUESTIONS_PER_LABEL="${POPE_QUESTIONS_PER_LABEL:-3}"
POPE_SEED="${POPE_SEED:-42}"
POPE_PARALLEL_WORKERS_WAS_SET="${POPE_PARALLEL_WORKERS+x}"
POPE_PARALLEL_WORKERS="${POPE_PARALLEL_WORKERS:-64}"
POPE_MAX_NEW_TOKENS_WAS_SET="${POPE_MAX_NEW_TOKENS+x}"
POPE_MAX_NEW_TOKENS="${POPE_MAX_NEW_TOKENS:-5120}"
POPE_MAX_SAMPLES="${POPE_MAX_SAMPLES:-0}"
POPE_USE_PREPARED_QUERY="${POPE_USE_PREPARED_QUERY:-False}"
AMBER_ROOT="${AMBER_ROOT:-${BENCHMARK_DATA_DIR:-/home/liuyanlin.lyl/notebook/data}/AMBER}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-}"
AMBER_EVAL_TYPE="${AMBER_EVAL_TYPE:-a}"
AMBER_MAX_SAMPLES="${AMBER_MAX_SAMPLES:-0}"
AMBER_PARALLEL_WORKERS_WAS_SET="${AMBER_PARALLEL_WORKERS+x}"
AMBER_PARALLEL_WORKERS="${AMBER_PARALLEL_WORKERS:-64}"
AMBER_OFFICIAL_EVAL_WORKERS="${AMBER_OFFICIAL_EVAL_WORKERS:-16}"
AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET="${AMBER_MAX_NEW_TOKENS_GENERATIVE+x}"
AMBER_MAX_NEW_TOKENS_GENERATIVE="${AMBER_MAX_NEW_TOKENS_GENERATIVE:-5120}"
AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET="${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE+x}"
AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-5120}"
AMBER_SKIP_OFFICIAL_EVAL="${AMBER_SKIP_OFFICIAL_EVAL:-False}"
AMBER_WORD_ASSOCIATION="${AMBER_WORD_ASSOCIATION:-}"
AMBER_SAFE_WORDS="${AMBER_SAFE_WORDS:-}"
AMBER_ANNOTATION="${AMBER_ANNOTATION:-}"
AMBER_METRICS="${AMBER_METRICS:-}"
CHAIR_MAX_NEW_TOKENS_WAS_SET="${CHAIR_MAX_NEW_TOKENS+x}"
CHAIR_MAX_NEW_TOKENS="${CHAIR_MAX_NEW_TOKENS:-5120}"
CHAIR_PARALLEL_WORKERS_WAS_SET="${CHAIR_PARALLEL_WORKERS+x}"
CHAIR_PARALLEL_WORKERS="${CHAIR_PARALLEL_WORKERS:-8}"
CHAIR_MAX_SAMPLES="${CHAIR_MAX_SAMPLES:-0}"
CHAIR_SAVE_LOGPROBS="${CHAIR_SAVE_LOGPROBS:-False}"
CHAIR_TOP_LOGPROBS="${CHAIR_TOP_LOGPROBS:-5}"
EVAL_OPD_TRACE="${EVAL_OPD_TRACE:-False}"
EVAL_OPD_TRACE_TOPK="${EVAL_OPD_TRACE_TOPK:-50}"
EVAL_OPD_TRACE_ENTROPY="${EVAL_OPD_TRACE_ENTROPY:-True}"
EVAL_OPD_TRACE_SCORE_BASELINE="${EVAL_OPD_TRACE_SCORE_BASELINE:-False}"
EVAL_OPD_TRACE_CASE_ANALYSIS="${EVAL_OPD_TRACE_CASE_ANALYSIS:-}"
EVAL_OPD_TRACE_MAX_SAMPLES="${EVAL_OPD_TRACE_MAX_SAMPLES:-0}"
VISION_BENCHMARK="${VISION_BENCHMARK:-mmstar}"
VISION_BENCHMARK_DATA_DIR="${VISION_BENCHMARK_DATA_DIR:-${BENCHMARK_DATA_DIR:-/home/liuyanlin.lyl/notebook/data}}"
VISION_BENCHMARK_AUTO_DOWNLOAD="${VISION_BENCHMARK_AUTO_DOWNLOAD:-${BENCHMARK_AUTO_DOWNLOAD:-True}}"
VISION_BENCHMARK_CLEAN_SOURCE="${VISION_BENCHMARK_CLEAN_SOURCE:-${BENCHMARK_CLEAN_SOURCE:-False}}"
VISION_BENCHMARK_REFRESH_PROMPTS_WAS_SET="${VISION_BENCHMARK_REFRESH_PROMPTS+x}${BENCHMARK_REFRESH_PROMPTS+x}"
VISION_BENCHMARK_REFRESH_PROMPTS="${VISION_BENCHMARK_REFRESH_PROMPTS:-${BENCHMARK_REFRESH_PROMPTS:-False}}"
VISION_BENCHMARK_OUTPUT_SUFFIX="${VISION_BENCHMARK_OUTPUT_SUFFIX:-${BENCHMARK_OUTPUT_SUFFIX:-}}"
VISION_MAX_TOKENS_WAS_SET="${VISION_MAX_TOKENS+x}"
VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-5120}"
VISION_PARALLEL_WORKERS_WAS_SET="${VISION_PARALLEL_WORKERS+x}"
VISION_PARALLEL_WORKERS="${VISION_PARALLEL_WORKERS:-128}"
VISION_MAX_RETRIES="${VISION_MAX_RETRIES:-3}"
VISION_ENABLE_THINKING="${VISION_ENABLE_THINKING:-}"
# Use non-empty check instead of +x to avoid bash version quirks with set -u
if [[ -n "${ENABLE_THINKING:-}" || -n "${VISION_ENABLE_THINKING:-}" ]]; then
    ENABLE_THINKING_WAS_SET="yes"
else
    ENABLE_THINKING_WAS_SET=""
fi
ENABLE_THINKING="${ENABLE_THINKING:-${VISION_ENABLE_THINKING:-}}"
FINAL_ANSWER_ONLY_WAS_SET="${FINAL_ANSWER_ONLY+x}"
FINAL_ANSWER_ONLY="${FINAL_ANSWER_ONLY:-}"
RULE_ONLY_JUDGE_WAS_SET="${RULE_ONLY_JUDGE+x}"
RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-False}"
MCQ_EXTRACT_MODE_WAS_SET="${MCQ_EXTRACT_MODE+x}"
MCQ_EXTRACT_MODE="${MCQ_EXTRACT_MODE:-legacy}"

normalize_eval_mode() {
    local mode
    local token
    local normalized=()
    mode="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    IFS=',' read -ra tokens <<< "$mode"
    for token in "${tokens[@]}"; do
        case "$token" in
            frequent|coco)
                normalized+=(chair pope)
                ;;
            all)
                normalized+=(chair pope vision amber)
                ;;
            cvbench|cv_bench|cv-bench)
                normalized+=(cv-bench)
                ;;
            mmstar|chair|pope|vision|amber)
                normalized+=("$token")
                ;;
            "")
                ;;
            *)
                echo "Error: unsupported eval task '$token'. Use chair,pope,mmstar,cv-bench,vision,amber,frequent,all." >&2
                exit 1
                ;;
        esac
    done

    local seen=","
    local unique=()
    for token in "${normalized[@]}"; do
        if [[ "$seen" != *",$token,"* ]]; then
            unique+=("$token")
            seen+="$token,"
        fi
    done
    local joined
    joined="$(IFS=','; echo "${unique[*]}")"
    echo "$joined"
}

has_eval_task() {
    local mode="$1"
    local task="$2"
    [[ ",${mode}," == *",${task},"* ]]
}

append_csv_unique() {
    local current="$1"
    local item="$2"
    [[ -z "$item" ]] && {
        echo "$current"
        return 0
    }
    if [[ ",${current}," == *",${item},"* ]]; then
        echo "$current"
    elif [[ -z "$current" ]]; then
        echo "$item"
    else
        echo "${current},${item}"
    fi
}

resolve_vision_benchmarks() {
    local mode="$1"
    local benches=""
    if has_eval_task "$mode" "vision"; then
        benches="$VISION_BENCHMARK"
    fi
    if has_eval_task "$mode" "mmstar"; then
        benches="$(append_csv_unique "$benches" "mmstar")"
    fi
    if has_eval_task "$mode" "cv-bench"; then
        benches="$(append_csv_unique "$benches" "cv-bench")"
    fi
    echo "$benches"
}

has_vision_eval_task() {
    local mode="$1"
    has_eval_task "$mode" "vision" || has_eval_task "$mode" "mmstar" || has_eval_task "$mode" "cv-bench"
}

infer_model_profile() {
    local explicit_profile="$1"
    local path_lc
    if [[ -n "$explicit_profile" ]]; then
        echo "$explicit_profile"
        return 0
    fi
    path_lc="$(echo "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')"
    if [[ "$path_lc" == *"thinking"* ]]; then
        if [[ "$path_lc" == *"8b"* ]]; then
            echo "qwen3vl_8b_thinking"
        elif [[ "$path_lc" == *"2b"* ]]; then
            echo "qwen3vl_2b_thinking"
        else
            echo "generic_thinking"
        fi
    elif [[ "$path_lc" == *"instruct"* ]]; then
        echo "qwen3vl_instruct"
    else
        echo "default"
    fi
}

resolve_result_version_tag() {
    local profile="$1"
    local requested_tag="$2"
    if [[ -n "$RESULT_VERSION_TAG" ]]; then
        echo "$RESULT_VERSION_TAG"
        return 0
    fi
    case "$profile" in
        qwen3vl_2b_thinking|qwen3vl_8b_thinking|generic_thinking)
            echo "thinking"
            ;;
        qwen3vl_instruct)
            echo "instruct"
            ;;
        *)
            echo "$requested_tag"
            ;;
    esac
}

apply_model_profile_defaults() {
    local profile="$1"
    case "$profile" in
        qwen3vl_2b_thinking)
            [[ -z "$ENABLE_THINKING_WAS_SET" ]] && ENABLE_THINKING="True"
            [[ -z "$FINAL_ANSWER_ONLY_WAS_SET" ]] && FINAL_ANSWER_ONLY="True"
            [[ -z "$CLEANUP_LOCAL_CKPT_WAS_SET" ]] && CLEANUP_LOCAL_CKPT="False"
            [[ -z "$DEGRADATION_MODE_WAS_SET" ]] && DEGRADATION_MODE="original"
            [[ -z "$STUDENT_RATIO_WAS_SET" ]] && STUDENT_RATIO="1.0"
            [[ -z "$TARGET_PX_WAS_SET" ]] && TARGET_PX="448"
            [[ -z "$VLLM_CUSTOM_OPS_WAS_SET" ]] && VLLM_CUSTOM_OPS="none"
            [[ -z "$VLLM_TENSOR_PARALLEL_SIZE_WAS_SET" ]] && VLLM_TENSOR_PARALLEL_SIZE="1"
            [[ -z "$VLLM_MAX_MODEL_LEN_WAS_SET" ]] && VLLM_MAX_MODEL_LEN="${THINKING_2B_VLLM_MAX_MODEL_LEN:-32768}"
            [[ -z "$VLLM_MAX_NUM_SEQS_WAS_SET" ]] && VLLM_MAX_NUM_SEQS="${THINKING_2B_VLLM_MAX_NUM_SEQS:-32}"
            [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]] && VISION_MAX_TOKENS="${THINKING_2B_VISION_MAX_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]] && VISION_MAX_TOKENS_WAS_SET="profile"
            [[ -z "$VISION_PARALLEL_WORKERS_WAS_SET" ]] && VISION_PARALLEL_WORKERS="${THINKING_2B_VISION_PARALLEL_WORKERS:-32}"
            [[ -z "$POPE_MAX_NEW_TOKENS_WAS_SET" ]] && POPE_MAX_NEW_TOKENS="${THINKING_2B_POPE_MAX_NEW_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$POPE_PARALLEL_WORKERS_WAS_SET" ]] && POPE_PARALLEL_WORKERS="${THINKING_2B_POPE_PARALLEL_WORKERS:-24}"
            [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]] && AMBER_MAX_NEW_TOKENS_GENERATIVE="${THINKING_2B_AMBER_MAX_NEW_TOKENS_GENERATIVE:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]] && AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${THINKING_2B_AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$AMBER_PARALLEL_WORKERS_WAS_SET" ]] && AMBER_PARALLEL_WORKERS="${THINKING_2B_AMBER_PARALLEL_WORKERS:-8}"
            [[ -z "$CHAIR_MAX_NEW_TOKENS_WAS_SET" ]] && CHAIR_MAX_NEW_TOKENS="${THINKING_2B_CHAIR_MAX_NEW_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$CHAIR_PARALLEL_WORKERS_WAS_SET" ]] && CHAIR_PARALLEL_WORKERS="${THINKING_2B_CHAIR_PARALLEL_WORKERS:-4}"
            ;;
        qwen3vl_8b_thinking)
            [[ -z "$ENABLE_THINKING_WAS_SET" ]] && ENABLE_THINKING="True"
            [[ -z "$FINAL_ANSWER_ONLY_WAS_SET" ]] && FINAL_ANSWER_ONLY="True"
            [[ -z "$CLEANUP_LOCAL_CKPT_WAS_SET" ]] && CLEANUP_LOCAL_CKPT="False"
            [[ -z "$DEGRADATION_MODE_WAS_SET" ]] && DEGRADATION_MODE="original"
            [[ -z "$STUDENT_RATIO_WAS_SET" ]] && STUDENT_RATIO="1.0"
            [[ -z "$TARGET_PX_WAS_SET" ]] && TARGET_PX="448"
            [[ -z "$VLLM_CUSTOM_OPS_WAS_SET" ]] && VLLM_CUSTOM_OPS="none"
            [[ -z "$VLLM_TENSOR_PARALLEL_SIZE_WAS_SET" ]] && VLLM_TENSOR_PARALLEL_SIZE="8"
            [[ -z "$VLLM_MAX_MODEL_LEN_WAS_SET" ]] && VLLM_MAX_MODEL_LEN="${THINKING_8B_VLLM_MAX_MODEL_LEN:-12288}"
            [[ -z "$VLLM_MAX_NUM_SEQS_WAS_SET" ]] && VLLM_MAX_NUM_SEQS="${THINKING_8B_VLLM_MAX_NUM_SEQS:-8}"
            [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]] && VISION_MAX_TOKENS="${THINKING_8B_VISION_MAX_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]] && VISION_MAX_TOKENS_WAS_SET="profile"
            [[ -z "$VISION_PARALLEL_WORKERS_WAS_SET" ]] && VISION_PARALLEL_WORKERS="${THINKING_8B_VISION_PARALLEL_WORKERS:-16}"
            [[ -z "$POPE_MAX_NEW_TOKENS_WAS_SET" ]] && POPE_MAX_NEW_TOKENS="${THINKING_8B_POPE_MAX_NEW_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$POPE_PARALLEL_WORKERS_WAS_SET" ]] && POPE_PARALLEL_WORKERS="${THINKING_8B_POPE_PARALLEL_WORKERS:-8}"
            [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]] && AMBER_MAX_NEW_TOKENS_GENERATIVE="${THINKING_8B_AMBER_MAX_NEW_TOKENS_GENERATIVE:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]] && AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${THINKING_8B_AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$AMBER_PARALLEL_WORKERS_WAS_SET" ]] && AMBER_PARALLEL_WORKERS="${THINKING_8B_AMBER_PARALLEL_WORKERS:-4}"
            [[ -z "$CHAIR_MAX_NEW_TOKENS_WAS_SET" ]] && CHAIR_MAX_NEW_TOKENS="${THINKING_8B_CHAIR_MAX_NEW_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$CHAIR_PARALLEL_WORKERS_WAS_SET" ]] && CHAIR_PARALLEL_WORKERS="${THINKING_8B_CHAIR_PARALLEL_WORKERS:-2}"
            ;;
        generic_thinking)
            [[ -z "$ENABLE_THINKING_WAS_SET" ]] && ENABLE_THINKING="True"
            [[ -z "$FINAL_ANSWER_ONLY_WAS_SET" ]] && FINAL_ANSWER_ONLY="True"
            [[ -z "$CLEANUP_LOCAL_CKPT_WAS_SET" ]] && CLEANUP_LOCAL_CKPT="False"
            [[ -z "$DEGRADATION_MODE_WAS_SET" ]] && DEGRADATION_MODE="original"
            [[ -z "$STUDENT_RATIO_WAS_SET" ]] && STUDENT_RATIO="1.0"
            [[ -z "$TARGET_PX_WAS_SET" ]] && TARGET_PX="448"
            [[ -z "$VLLM_CUSTOM_OPS_WAS_SET" ]] && VLLM_CUSTOM_OPS="none"
            [[ -z "$VLLM_TENSOR_PARALLEL_SIZE_WAS_SET" ]] && VLLM_TENSOR_PARALLEL_SIZE="8"
            [[ -z "$VLLM_MAX_MODEL_LEN_WAS_SET" ]] && VLLM_MAX_MODEL_LEN="${THINKING_VLLM_MAX_MODEL_LEN:-12288}"
            [[ -z "$VLLM_MAX_NUM_SEQS_WAS_SET" ]] && VLLM_MAX_NUM_SEQS="${THINKING_VLLM_MAX_NUM_SEQS:-8}"
            [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]] && VISION_MAX_TOKENS="${THINKING_VISION_MAX_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]] && VISION_MAX_TOKENS_WAS_SET="profile"
            [[ -z "$VISION_PARALLEL_WORKERS_WAS_SET" ]] && VISION_PARALLEL_WORKERS="${THINKING_VISION_PARALLEL_WORKERS:-16}"
            [[ -z "$POPE_MAX_NEW_TOKENS_WAS_SET" ]] && POPE_MAX_NEW_TOKENS="${THINKING_POPE_MAX_NEW_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$POPE_PARALLEL_WORKERS_WAS_SET" ]] && POPE_PARALLEL_WORKERS="${THINKING_POPE_PARALLEL_WORKERS:-8}"
            [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]] && AMBER_MAX_NEW_TOKENS_GENERATIVE="${THINKING_AMBER_MAX_NEW_TOKENS_GENERATIVE:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]] && AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${THINKING_AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$AMBER_PARALLEL_WORKERS_WAS_SET" ]] && AMBER_PARALLEL_WORKERS="${THINKING_AMBER_PARALLEL_WORKERS:-4}"
            [[ -z "$CHAIR_MAX_NEW_TOKENS_WAS_SET" ]] && CHAIR_MAX_NEW_TOKENS="${THINKING_CHAIR_MAX_NEW_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
            [[ -z "$CHAIR_PARALLEL_WORKERS_WAS_SET" ]] && CHAIR_PARALLEL_WORKERS="${THINKING_CHAIR_PARALLEL_WORKERS:-2}"
            ;;
        qwen3vl_instruct|default)
            [[ -z "$VLLM_TENSOR_PARALLEL_SIZE_WAS_SET" ]] && VLLM_TENSOR_PARALLEL_SIZE="1"
            # Keep historical defaults unless the caller explicitly overrides env vars.
            ;;
        *)
            echo "Error: unsupported MODEL_PROFILE=${profile}. Use default, qwen3vl_instruct, qwen3vl_2b_thinking, qwen3vl_8b_thinking, or generic_thinking." >&2
            exit 1
            ;;
    esac
    return 0
}

apply_official_aux_defaults() {
    local benches="$1"
    if [[ ",${benches}," == *",mmstar,"* || ",${benches}," == *",cv-bench,"* ]]; then
        local default_max_tokens="${EVAL_MAX_TOKENS:-5120}"
        if [[ "${ENABLE_THINKING}" == "True" || "${ENABLE_THINKING}" == "true" || "${ENABLE_THINKING}" == "1" ]]; then
            default_max_tokens="${VISION_THINKING_MAX_TOKENS:-${EVAL_MAX_TOKENS:-5120}}"
        fi
        if [[ -z "$VISION_BENCHMARK_REFRESH_PROMPTS_WAS_SET" ]]; then
            VISION_BENCHMARK_REFRESH_PROMPTS="True"
        fi
        if [[ -z "$RULE_ONLY_JUDGE_WAS_SET" ]]; then
            RULE_ONLY_JUDGE="True"
        fi
        if [[ -z "$MCQ_EXTRACT_MODE_WAS_SET" ]]; then
            MCQ_EXTRACT_MODE="official"
        fi
        if [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]]; then
            VISION_MAX_TOKENS="$default_max_tokens"
        fi
    fi
    return 0
}

is_truthy() {
    [[ "$1" == "True" || "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "Y" || "$1" == "y" ]]
}

realpath_for_cleanup() {
    "$PYTHON_BIN" - "$1" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
}

safe_remove_checkpoint_dir() {
    local target="$1"
    local base_dir="${RES_OPD_ROOT}/checkpoints"
    local target_real
    local base_real
    target_real="$(realpath_for_cleanup "$target")"
    base_real="$(realpath_for_cleanup "$base_dir")"

    if [[ "$target_real" == "$base_real"/* ]]; then
        rm -rf "$target"
        return 0
    fi

    if is_truthy "$ALLOW_CLEANUP_OUTSIDE_CKPT_BASE"; then
        echo "  WARNING: removing checkpoint outside ${base_real}: ${target_real}" >&2
        rm -rf "$target"
        return 0
    fi

    echo "  WARNING: refusing to remove path outside ${base_real}: ${target_real}" >&2
    echo "  Set ALLOW_CLEANUP_OUTSIDE_CKPT_BASE=True only if this is intentional." >&2
    return 1
}

get_oss_name_from_exp_name() {
    local exp_name="$1"
    local suffix
    if [[ "$exp_name" == ResOPD_* ]]; then
        echo "$exp_name"
        return 0
    elif [[ "$exp_name" == Res-OPD-Qwen3VL-2B-Instruct-* ]]; then
        # Backward-compatible path for existing Instruct checkpoints.
        suffix="${exp_name#Res-OPD-Qwen3VL-2B-Instruct-}"
    elif [[ "$exp_name" == Res-OPD-* ]]; then
        suffix="${exp_name#Res-OPD-}"
    else
        suffix="$exp_name"
    fi

    local epoch_tag=""
    if [[ "$suffix" =~ ^(.+)-(e[0-9]+)$ ]]; then
        suffix="${BASH_REMATCH[1]}"
        epoch_tag="-${BASH_REMATCH[2]}"
    fi

    echo "ResOPD_${suffix//-/_}${epoch_tag}"
}

port_listener_pids() {
    local pids=""
    if command -v lsof >/dev/null 2>&1; then
        pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"
    elif command -v fuser >/dev/null 2>&1; then
        pids="$(fuser -n tcp "$PORT" 2>/dev/null || true)"
    elif command -v ss >/dev/null 2>&1; then
        pids="$(ss -ltnp "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' || true)"
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

is_vllm_process_on_port() {
    local cmd="$1"
    [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* ]] || {
        [[ "$cmd" == *"vllm"* && "$cmd" == *"--port ${PORT}"* ]]
    }
}

ensure_vllm_port_available() {
    local pids
    pids="$(port_listener_pids)"
    if [[ -z "$pids" ]]; then
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "Error: port ${PORT} has a live /health endpoint, but no PID could be detected." >&2
            echo "Install lsof/fuser/ss or set a different VLLM_PORT." >&2
            exit 1
        fi
        return 0
    fi

    echo "  Port ${PORT} is already in use by: ${pids}"
    if ! is_truthy "$VLLM_PORT_CLEANUP"; then
        echo "Error: VLLM_PORT_CLEANUP=False and port ${PORT} is busy." >&2
        exit 1
    fi

    local pid
    local cmd
    local blocked=0
    for pid in $pids; do
        cmd="$(pid_cmdline "$pid")"
        if is_vllm_process_on_port "$cmd"; then
            echo "  Killing stale vLLM process pid=${pid}: ${cmd}"
            kill "$pid" 2>/dev/null || true
        else
            echo "Error: port ${PORT} is used by a non-vLLM process pid=${pid}: ${cmd}" >&2
            blocked=1
        fi
    done
    if [[ "$blocked" -ne 0 ]]; then
        exit 1
    fi

    for _ in $(seq 1 "$VLLM_PORT_CLEANUP_WAIT"); do
        pids="$(port_listener_pids)"
        [[ -z "$pids" ]] && return 0
        sleep 1
    done

    pids="$(port_listener_pids)"
    if [[ -n "$pids" ]]; then
        echo "  Port ${PORT} still busy after graceful cleanup; force killing stale vLLM pids: ${pids}"
        for pid in $pids; do
            cmd="$(pid_cmdline "$pid")"
            if is_vllm_process_on_port "$cmd"; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    fi
}

vision_benchmark_json_name() {
    case "$1" in
        zoombench) echo "zoombench.json" ;;
        vstar) echo "vstar.json" ;;
        hrbench-4k) echo "hr_bench_4k.json" ;;
        hrbench-8k) echo "hr_bench_8k.json" ;;
        mme-realworld) echo "MME_RealWorld.json" ;;
        mme-realworld-cn) echo "MME_RealWorld_CN.json" ;;
        mme-realworld-lite) echo "MME_RealWorld_Lite.json" ;;
        mmstar) echo "mmstar.json" ;;
        pope) echo "POPE.json" ;;
        pope_adv) echo "POPE_adv.json" ;;
        pope_pop) echo "POPE_pop.json" ;;
        pope_random) echo "POPE_random.json" ;;
        cv-bench) echo "cv_bench.json" ;;
        mmvp) echo "mmvp.json" ;;
        visualprobe) echo "visualprobe.json" ;;
        *)
            echo ""
            ;;
    esac
}

prepare_vision_benchmark_data() {
    if ! has_vision_eval_task "$EVAL_MODE"; then
        return 0
    fi

    echo ""
    echo "[preflight] Preparing Vision-OPD benchmark data before vLLM ..."
    mkdir -p "$VISION_BENCHMARK_DATA_DIR"

    local old_ifs="$IFS"
    local benchmarks=()
    local bench
    IFS=',' read -r -a benchmarks <<< "$EFFECTIVE_VISION_BENCHMARK"
    IFS="$old_ifs"

    for bench in "${benchmarks[@]}"; do
        bench="$(echo "$bench" | xargs)"
        [[ -z "$bench" ]] && continue

        local bench_json
        bench_json="$(vision_benchmark_json_name "$bench")"
        if [[ -z "$bench_json" ]]; then
            echo "Error: unsupported Vision-OPD benchmark: $bench" >&2
            exit 1
        fi

        local benchmark_json_path="${VISION_BENCHMARK_DATA_DIR}/${bench_json}"
        local prepare_args=(
            --benchmark "$bench"
            --data_dir "$VISION_BENCHMARK_DATA_DIR"
        )
        if is_truthy "$VISION_BENCHMARK_REFRESH_PROMPTS"; then
            prepare_args+=(--refresh-prompts)
        fi
        if is_truthy "$VISION_BENCHMARK_CLEAN_SOURCE"; then
            prepare_args+=(--clean-source)
        fi

        if [[ -s "$benchmark_json_path" ]]; then
            echo "  Found ${bench}: ${benchmark_json_path}"
            if is_truthy "$VISION_BENCHMARK_REFRESH_PROMPTS" || is_truthy "$VISION_BENCHMARK_CLEAN_SOURCE"; then
                "$PYTHON_BIN" "${VISION_OPD_ROOT}/eval/prepare_data.py" "${prepare_args[@]}"
            fi
        else
            if ! is_truthy "$VISION_BENCHMARK_AUTO_DOWNLOAD"; then
                echo "Error: benchmark JSON missing and auto-download is disabled: ${benchmark_json_path}" >&2
                exit 1
            fi
            echo "  Preparing ${bench}: ${benchmark_json_path}"
            "$PYTHON_BIN" "${VISION_OPD_ROOT}/eval/prepare_data.py" "${prepare_args[@]}"
        fi

        if [[ ! -s "$benchmark_json_path" ]]; then
            echo "Error: prepared benchmark JSON is missing or empty: ${benchmark_json_path}" >&2
            exit 1
        fi
    done
}

EVAL_MODE="$(normalize_eval_mode "$EVAL_MODE")"
EFFECTIVE_VISION_BENCHMARK="$(resolve_vision_benchmarks "$EVAL_MODE")"
MODEL_PROFILE="$(infer_model_profile "$MODEL_PROFILE")"
apply_model_profile_defaults "$MODEL_PROFILE"
apply_official_aux_defaults "$EFFECTIVE_VISION_BENCHMARK"
RESULT_VERSION_TAG="$(resolve_result_version_tag "$MODEL_PROFILE" "$VERSION_TAG")"
if [[ -z "$FINAL_ANSWER_ONLY" && ( "$ENABLE_THINKING" == "True" || "$ENABLE_THINKING" == "true" || "$ENABLE_THINKING" == "1" ) ]]; then
    FINAL_ANSWER_ONLY="True"
fi
if [[ "$FINAL_ANSWER_ONLY" == "True" || "$FINAL_ANSWER_ONLY" == "true" || "$FINAL_ANSWER_ONLY" == "1" ]]; then
    if [[ -z "$POPE_MAX_NEW_TOKENS_WAS_SET" && "$POPE_MAX_NEW_TOKENS" == "16" ]]; then
        POPE_MAX_NEW_TOKENS="${POPE_THINKING_MAX_NEW_TOKENS:-512}"
    fi
    if [[ -z "$CHAIR_MAX_NEW_TOKENS_WAS_SET" && "$CHAIR_MAX_NEW_TOKENS" == "384" ]]; then
        CHAIR_MAX_NEW_TOKENS="${CHAIR_THINKING_MAX_NEW_TOKENS:-2048}"
    fi
fi
if ! has_eval_task "$EVAL_MODE" "chair" && ! has_eval_task "$EVAL_MODE" "pope" && ! has_eval_task "$EVAL_MODE" "amber" && ! has_vision_eval_task "$EVAL_MODE"; then
    echo "Error: eval_mode must include chair, pope, mmstar, cv-bench, vision, amber, frequent, or all. Got: $EVAL_MODE" >&2
    exit 1
fi

# Determine eval output directory:
#   res-opd/eval_results/<version_tag>/<experiment_name>_<step_tag>/<dataset_tag>/
CKPT_ROOT="$MODEL_PATH"
STEP_TAG=""
if [[ "$(basename "$MODEL_PATH")" =~ ^global_step_[0-9]+$ ]]; then
    STEP_TAG="$(basename "$MODEL_PATH")"
    CKPT_ROOT="$(dirname "$MODEL_PATH")"
fi
EXPERIMENT_NAME="$(basename "$CKPT_ROOT")"
# Append step tag to distinguish different checkpoints of the same experiment
if [[ -n "$STEP_TAG" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${STEP_TAG}"
fi

infer_eval_spec() {
    local exp_name="$1"
    local inferred_mode="square"
    local inferred_student_px="0"
    local inferred_teacher_px="$TARGET_PX"
    local inferred_student_ratio="1.0"
    local inferred_teacher_ratio="1.0"
    if [[ "$exp_name" =~ -orig-sr([0-9.]+)-tr([0-9.]+) ]]; then
        inferred_mode="original"
        inferred_student_px="0"
        inferred_student_ratio="${BASH_REMATCH[1]}"
        inferred_teacher_ratio="${BASH_REMATCH[2]}"
    elif [[ "$exp_name" =~ -s([0-9]+)-t([0-9]+) ]]; then
        inferred_mode="square"
        inferred_student_px="${BASH_REMATCH[1]}"
        inferred_teacher_px="${BASH_REMATCH[2]}"
    elif [[ "$exp_name" =~ -s([0-9]+)(-|_) ]]; then
        inferred_mode="square"
        inferred_student_px="${BASH_REMATCH[1]}"
    fi
    DEGRADATION_MODE="${DEGRADATION_MODE:-$inferred_mode}"
    STUDENT_PX="${STUDENT_PX:-$inferred_student_px}"
    TEACHER_PX="${TEACHER_PX:-$inferred_teacher_px}"
    STUDENT_RATIO="${STUDENT_RATIO:-$inferred_student_ratio}"
    TEACHER_RATIO="${TEACHER_RATIO:-$inferred_teacher_ratio}"
}

infer_eval_spec "$EXPERIMENT_NAME"

case "$DEGRADATION_MODE" in
    square|original)
        ;;
    *)
        echo "Error: DEGRADATION_MODE must be square or original. Got: $DEGRADATION_MODE" >&2
        exit 1
        ;;
esac

# Build dataset tag from actual data file sizes
if [[ "$DATASET_VERSION" == "full" ]]; then
    TRAIN_FILE="${RES_OPD_ROOT}/data/train_5k.parquet"
    TEST_FILE="${RES_OPD_ROOT}/data/test_1000.json"
else
    TRAIN_FILE="${RES_OPD_ROOT}/data/train.parquet"
    TEST_FILE="${RES_OPD_ROOT}/data/test.json"
fi
DATASET_TAG=""
if [[ -f "$TRAIN_FILE" && -f "$TEST_FILE" ]]; then
    TRAIN_N=$(/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))" 2>/dev/null || echo "?")
    TEST_N=$(/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 -c "import json; print(len(json.load(open('$TEST_FILE'))))" 2>/dev/null || echo "?")
    DATASET_TAG="train${TRAIN_N}_test${TEST_N}"
else
    DATASET_TAG="unknown_dataset"
fi
if [[ "$DEGRADATION_MODE" == "original" ]]; then
    RATIO_TAG="${STUDENT_RATIO//./p}"
    DATASET_TAG="${DATASET_TAG}_original_sr${RATIO_TAG}"
fi
# When DATASET_VERSION=full, add a "full/" parent folder for separation.
# Legacy results stay directly under eval_results/<version_tag>/<exp_name>/<dataset_tag>/
if [[ "$DATASET_VERSION" == "full" ]]; then
    OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${RESULT_VERSION_TAG}/full/${EXPERIMENT_NAME}/${DATASET_TAG}"
else
    OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${RESULT_VERSION_TAG}/${EXPERIMENT_NAME}/${DATASET_TAG}"
fi
if [[ -n "$EVAL_OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$EVAL_OUTPUT_DIR"
fi

if has_eval_task "$EVAL_MODE" "chair" && [ ! -f "$TEST_JSON" ]; then
    echo "Error: test.json not found at $TEST_JSON" >&2
    echo "Run prepare_data.py first." >&2
    exit 1
fi

echo "============================================================"
echo " Res-OPD Evaluation"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "Deg mode:    $DEGRADATION_MODE"
echo "Student px:  $STUDENT_PX (0 = original image)"
echo "Teacher px:  $TEACHER_PX (square-mode offline OPD trace)"
echo "Target px:   $TARGET_PX"
echo "Student ratio: $STUDENT_RATIO (original mode)"
echo "Teacher ratio: $TEACHER_RATIO (original-mode offline OPD trace)"
echo "Eval mode:   $EVAL_MODE"
if has_eval_task "$EVAL_MODE" "chair"; then
    echo "CHAIR data:  $TEST_JSON"
    echo "CHAIR max tokens/workers: ${CHAIR_MAX_NEW_TOKENS}/${CHAIR_PARALLEL_WORKERS}"
    echo "CHAIR logprobs: ${CHAIR_SAVE_LOGPROBS} (top=${CHAIR_TOP_LOGPROBS})"
    echo "OPD eval trace: ${EVAL_OPD_TRACE} (topk=${EVAL_OPD_TRACE_TOPK}, entropy=${EVAL_OPD_TRACE_ENTROPY})"
fi
if has_eval_task "$EVAL_MODE" "pope"; then
    echo "POPE:        $POPE_BENCHMARK"
    echo "POPE source: $POPE_SOURCE"
    echo "POPE max tokens/workers: ${POPE_MAX_NEW_TOKENS}/${POPE_PARALLEL_WORKERS}"
fi
if has_eval_task "$EVAL_MODE" "amber"; then
    echo "AMBER root:  $AMBER_ROOT"
    echo "AMBER image: ${AMBER_IMAGE_ROOT:-<auto>}"
    echo "AMBER type:  $AMBER_EVAL_TYPE"
    echo "AMBER max tokens: generative=${AMBER_MAX_NEW_TOKENS_GENERATIVE}, discriminative=${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE}"
    echo "AMBER workers: $AMBER_PARALLEL_WORKERS"
    echo "AMBER official eval workers: $AMBER_OFFICIAL_EVAL_WORKERS"
fi
if has_vision_eval_task "$EVAL_MODE"; then
    echo "Vision-OPD:  $EFFECTIVE_VISION_BENCHMARK"
    echo "Vision data: $VISION_BENCHMARK_DATA_DIR"
    echo "Vision auto download: $VISION_BENCHMARK_AUTO_DOWNLOAD"
    echo "Vision clean source:  $VISION_BENCHMARK_CLEAN_SOURCE"
    echo "Vision refresh prompts: $VISION_BENCHMARK_REFRESH_PROMPTS"
    echo "Vision output suffix: $VISION_BENCHMARK_OUTPUT_SUFFIX"
    echo "Vision max tokens/workers: ${VISION_MAX_TOKENS}/${VISION_PARALLEL_WORKERS}"
    echo "MCQ extract mode: $MCQ_EXTRACT_MODE"
fi
echo "Model profile: $MODEL_PROFILE"
echo "Version tag: $VERSION_TAG"
echo "Result tag:  $RESULT_VERSION_TAG"
echo "Enable thinking: ${ENABLE_THINKING:-<unset>}"
echo "Final-answer only: ${FINAL_ANSWER_ONLY:-False}"
echo "vLLM max len: $VLLM_MAX_MODEL_LEN"
echo "vLLM max seqs/batched tokens: ${VLLM_MAX_NUM_SEQS:-<default>}/${VLLM_MAX_NUM_BATCHED_TOKENS:-<default>}"
echo "Output:      $OUTPUT_DIR"
echo "Shard:       ${EVAL_SHARD_INDEX}/${EVAL_SHARD_COUNT}"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

prepare_vision_benchmark_data

# --- Step 1: Start vLLM server ---
echo ""
echo "[1/3] Starting vLLM server on port $PORT ..."
ensure_vllm_port_available
# Disable prometheus metrics to avoid '_IncludedRouter' compatibility issue with vLLM 0.18+
export VLLM_DISABLE_PROMETHEUS=1
vllm_args=(
    -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --port "$PORT"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --disable-frontend-multiprocessing
)
if [[ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]]; then
    vllm_args+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE")
fi
if [[ -n "$VLLM_MAX_NUM_SEQS" ]]; then
    vllm_args+=(--max-num-seqs "$VLLM_MAX_NUM_SEQS")
fi
if [[ -n "$VLLM_MAX_NUM_BATCHED_TOKENS" ]]; then
    vllm_args+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi
"$PYTHON_BIN" "${vllm_args[@]}" &
VLLM_PID=$!

cleanup() {
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo ""
        echo "[cleanup] Shutting down vLLM server ..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# Wait for server to be ready
echo "  Waiting for vLLM server (pid=$VLLM_PID) ..."
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
for i in $(seq 1 $VLLM_STARTUP_TIMEOUT); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "  vLLM server ready after ${i}s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "  vLLM server exited unexpectedly!" >&2
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "  vLLM server failed to start within ${VLLM_STARTUP_TIMEOUT}s" >&2
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# --- Step 2: Run evaluation ---
echo ""
echo "[2/3] Running selected evaluation(s) ..."
EVAL_FAILURES=0
if has_eval_task "$EVAL_MODE" "chair"; then
    echo "  Running CHAIR ..."
    chair_extra_args=(
        --max-new-tokens "$CHAIR_MAX_NEW_TOKENS"
        --parallel-workers "$CHAIR_PARALLEL_WORKERS"
        --max-samples "$CHAIR_MAX_SAMPLES"
    )
    if [[ "$CHAIR_SAVE_LOGPROBS" == "True" || "$CHAIR_SAVE_LOGPROBS" == "true" || "$CHAIR_SAVE_LOGPROBS" == "1" ]]; then
        chair_extra_args+=(--save-logprobs --top-logprobs "$CHAIR_TOP_LOGPROBS")
    fi
    if [[ -n "$ENABLE_THINKING" ]]; then
        chair_extra_args+=(--enable-thinking "$ENABLE_THINKING")
    fi
    if [[ "$FINAL_ANSWER_ONLY" == "True" || "$FINAL_ANSWER_ONLY" == "true" || "$FINAL_ANSWER_ONLY" == "1" ]]; then
        chair_extra_args+=(--final-answer-only)
    fi
    if ! "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_chair.py" \
        --api-base "http://localhost:$PORT/v1/" \
        --model-name "$MODEL_NAME" \
        --test-json "$TEST_JSON" \
        --output-dir "$OUTPUT_DIR" \
        --student-px "$STUDENT_PX" \
        --target-px "$TARGET_PX" \
        --degradation-mode "$DEGRADATION_MODE" \
        --student-ratio "$STUDENT_RATIO" \
        --shard-count "$EVAL_SHARD_COUNT" \
        --shard-index "$EVAL_SHARD_INDEX" \
        ${chair_extra_args[@]+"${chair_extra_args[@]}"}; then
        echo "  WARNING: CHAIR failed; keeping any completed outputs." >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
fi

if has_eval_task "$EVAL_MODE" "pope"; then
    echo "  Running POPE ..."
    pope_extra_args=()
    if [[ "$POPE_USE_PREPARED_QUERY" == "True" || "$POPE_USE_PREPARED_QUERY" == "true" ]]; then
        pope_extra_args+=(--use-prepared-query)
    fi
    if [[ -n "$ENABLE_THINKING" ]]; then
        pope_extra_args+=(--enable-thinking "$ENABLE_THINKING")
    fi
    if ! "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_pope.py" \
        --api-base "http://localhost:$PORT/v1/" \
        --api-key "${OPENAI_API_KEY:-EMPTY}" \
        --model-name "$MODEL_NAME" \
        --benchmark "$POPE_BENCHMARK" \
        --pope-source "$POPE_SOURCE" \
        --test-json "$TEST_JSON" \
        --questions-per-label "$POPE_QUESTIONS_PER_LABEL" \
        --seed "$POPE_SEED" \
        --vision-opd-root "$VISION_OPD_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --student-px "$STUDENT_PX" \
        --target-px "$TARGET_PX" \
        --degradation-mode "$DEGRADATION_MODE" \
        --student-ratio "$STUDENT_RATIO" \
        --max-new-tokens "$POPE_MAX_NEW_TOKENS" \
        --max-samples "$POPE_MAX_SAMPLES" \
        --shard-count "$EVAL_SHARD_COUNT" \
        --shard-index "$EVAL_SHARD_INDEX" \
        --parallel-workers "$POPE_PARALLEL_WORKERS" \
        ${pope_extra_args[@]+"${pope_extra_args[@]}"}; then
        echo "  WARNING: POPE failed; keeping any completed outputs." >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
fi

if has_vision_eval_task "$EVAL_MODE"; then
    echo "  Running Vision-OPD benchmark(s) ..."
    if [[ "${RULE_ONLY_JUDGE}" != "True" && "${RULE_ONLY_JUDGE}" != "true" && -z "${JUDGE_API_BASE:-}" && -z "${JUDGE_MODEL_PATH:-}" ]]; then
        echo "  WARNING: Vision-OPD benchmarks require JUDGE_API_BASE/JUDGE_MODEL or JUDGE_MODEL_PATH. Skipping." >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    else
        vision_out_dir="${OUTPUT_DIR}"
        vision_judge_dir="${OUTPUT_DIR}"
        mkdir -p "$vision_out_dir" "$vision_judge_dir"
        vision_args=(
            API_BASE="http://localhost:$PORT/v1/"
            OPENAI_MODEL_ID="$MODEL_NAME"
            MODEL_NAME="${EXPERIMENT_NAME}"
            BENCHMARK="$EFFECTIVE_VISION_BENCHMARK"
            BENCHMARK_OUTPUT_LAYOUT="per_benchmark"
            BENCHMARK_DATA_DIR="$VISION_BENCHMARK_DATA_DIR"
            BENCHMARK_PREPARE_DATA=False
            BENCHMARK_AUTO_DOWNLOAD="$VISION_BENCHMARK_AUTO_DOWNLOAD"
            BENCHMARK_CLEAN_SOURCE="$VISION_BENCHMARK_CLEAN_SOURCE"
            BENCHMARK_REFRESH_PROMPTS="$VISION_BENCHMARK_REFRESH_PROMPTS"
            BENCHMARK_OUTPUT_SUFFIX="$VISION_BENCHMARK_OUTPUT_SUFFIX"
            OUT_DIR="$vision_out_dir"
            JUDGE_DIR="$vision_judge_dir"
            MAX_TOKENS="$VISION_MAX_TOKENS"
            MAX_RETRIES="$VISION_MAX_RETRIES"
            PARALLEL_WORKERS="$VISION_PARALLEL_WORKERS"
            RULE_ONLY_JUDGE="$RULE_ONLY_JUDGE"
            MCQ_EXTRACT_MODE="$MCQ_EXTRACT_MODE"
            BENCHMARK_SHARD_COUNT="$EVAL_SHARD_COUNT"
            BENCHMARK_SHARD_INDEX="$EVAL_SHARD_INDEX"
        )
        [[ -n "${OPENAI_API_KEY:-}" ]] && vision_args+=(OPENAI_API_KEY="$OPENAI_API_KEY")
        [[ -n "${JUDGE_API_BASE:-}" ]] && vision_args+=(JUDGE_API_BASE="$JUDGE_API_BASE")
        [[ -n "${JUDGE_API_KEY:-}" ]] && vision_args+=(JUDGE_API_KEY="$JUDGE_API_KEY")
        [[ -n "${JUDGE_MODEL:-}" ]] && vision_args+=(JUDGE_MODEL="$JUDGE_MODEL")
        [[ -n "${JUDGE_MODEL_PATH:-}" ]] && vision_args+=(JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH")
        [[ -n "${JUDGE_MAX_TOKENS:-}" ]] && vision_args+=(JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS")
        [[ -n "$ENABLE_THINKING" ]] && vision_args+=(ENABLE_THINKING="$ENABLE_THINKING")
        if ! env "${vision_args[@]}" bash "${VISION_OPD_ROOT}/eval/run_eval.sh"; then
            echo "  WARNING: Vision-OPD benchmark(s) failed; keeping any completed outputs." >&2
            EVAL_FAILURES=$((EVAL_FAILURES + 1))
        fi
    fi
fi

if has_eval_task "$EVAL_MODE" "amber"; then
    echo "  Running AMBER ..."
    amber_extra_args=()
    [[ -n "$AMBER_IMAGE_ROOT" ]] && amber_extra_args+=(--image-root "$AMBER_IMAGE_ROOT")
    if [[ "$AMBER_SKIP_OFFICIAL_EVAL" == "True" || "$AMBER_SKIP_OFFICIAL_EVAL" == "true" || "$AMBER_SKIP_OFFICIAL_EVAL" == "1" ]]; then
        amber_extra_args+=(--skip-official-eval)
    fi
    [[ -n "$AMBER_WORD_ASSOCIATION" ]] && amber_extra_args+=(--word-association "$AMBER_WORD_ASSOCIATION")
    [[ -n "$AMBER_SAFE_WORDS" ]] && amber_extra_args+=(--safe-words "$AMBER_SAFE_WORDS")
    [[ -n "$AMBER_ANNOTATION" ]] && amber_extra_args+=(--annotation "$AMBER_ANNOTATION")
    [[ -n "$AMBER_METRICS" ]] && amber_extra_args+=(--metrics "$AMBER_METRICS")
    [[ -n "$ENABLE_THINKING" ]] && amber_extra_args+=(--enable-thinking "$ENABLE_THINKING")
    if ! "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_amber.py" \
        --api-base "http://localhost:$PORT/v1/" \
        --api-key "${OPENAI_API_KEY:-EMPTY}" \
        --model-name "$MODEL_NAME" \
        --amber-root "$AMBER_ROOT" \
        --output-dir "$OUTPUT_DIR" \
        --student-px "$STUDENT_PX" \
        --target-px "$TARGET_PX" \
        --degradation-mode "$DEGRADATION_MODE" \
        --student-ratio "$STUDENT_RATIO" \
        --evaluation-type "$AMBER_EVAL_TYPE" \
        --max-new-tokens-generative "$AMBER_MAX_NEW_TOKENS_GENERATIVE" \
        --max-new-tokens-discriminative "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE" \
        --max-samples "$AMBER_MAX_SAMPLES" \
        --shard-count "$EVAL_SHARD_COUNT" \
        --shard-index "$EVAL_SHARD_INDEX" \
        --parallel-workers "$AMBER_PARALLEL_WORKERS" \
        --official-eval-workers "$AMBER_OFFICIAL_EVAL_WORKERS" \
        ${amber_extra_args[@]+"${amber_extra_args[@]}"}; then
        echo "  WARNING: AMBER failed; keeping any completed outputs." >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
fi

"$PYTHON_BIN" "${RES_OPD_ROOT}/eval/collect_benchmark_summary.py" \
    --output-dir "$OUTPUT_DIR" || true

# --- Step 3: Release vLLM before optional offline OPD scoring ---
echo ""
echo "[3/3] Shutting down vLLM server ..."
cleanup
trap - EXIT

if has_eval_task "$EVAL_MODE" "chair" && [[ "$EVAL_OPD_TRACE" == "True" || "$EVAL_OPD_TRACE" == "true" || "$EVAL_OPD_TRACE" == "1" ]]; then
    echo ""
    echo "[OPD trace] Forced-scoring eval captions ..."
    OPD_TRACE_JSONL="${OUTPUT_DIR}/opd_eval_trace.jsonl"
    OPD_TRACE_SUMMARY_JSON="${OUTPUT_DIR}/opd_eval_trace_summary.json"
    OPD_TRACE_SUMMARY_MD="${OUTPUT_DIR}/opd_eval_trace_summary.md"
    scorer_args=(
        --model-path "$MODEL_PATH"
        --eval-results "${OUTPUT_DIR}/eval_results.jsonl"
        --test-json "$TEST_JSON"
        --output-jsonl "$OPD_TRACE_JSONL"
        --degradation-mode "$DEGRADATION_MODE"
        --student-px "$STUDENT_PX"
        --teacher-px "$TEACHER_PX"
        --target-px "$TARGET_PX"
        --student-ratio "$STUDENT_RATIO"
        --teacher-ratio "$TEACHER_RATIO"
        --topk "$EVAL_OPD_TRACE_TOPK"
        --max-samples "$EVAL_OPD_TRACE_MAX_SAMPLES"
    )
    if [[ "$EVAL_OPD_TRACE_ENTROPY" == "True" || "$EVAL_OPD_TRACE_ENTROPY" == "true" || "$EVAL_OPD_TRACE_ENTROPY" == "1" ]]; then
        scorer_args+=(--entropy)
    fi
    if [[ "$EVAL_OPD_TRACE_SCORE_BASELINE" == "True" || "$EVAL_OPD_TRACE_SCORE_BASELINE" == "true" || "$EVAL_OPD_TRACE_SCORE_BASELINE" == "1" ]]; then
        scorer_args+=(--score-baseline-caption)
    fi
    if [[ -n "$EVAL_OPD_TRACE_CASE_ANALYSIS" ]]; then
        scorer_args+=(--case-analysis "$EVAL_OPD_TRACE_CASE_ANALYSIS")
    fi
    if ! "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/score_opd_eval_trace.py" "${scorer_args[@]}"; then
        echo "  WARNING: OPD eval trace scorer failed; keeping eval outputs." >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    else
        analyze_args=(
            --trace-dir "$OPD_TRACE_JSONL"
            --output-json "$OPD_TRACE_SUMMARY_JSON"
            --output-md "$OPD_TRACE_SUMMARY_MD"
        )
        if [[ -n "$EVAL_OPD_TRACE_CASE_ANALYSIS" ]]; then
            analyze_args+=(--case-analysis "$EVAL_OPD_TRACE_CASE_ANALYSIS")
        fi
        "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/analyze_opd_trace.py" "${analyze_args[@]}"
    fi
fi

# --- Step 4: Cleanup local checkpoint after verifying OSS backup ---
if [[ "$CLEANUP_LOCAL_CKPT" == "True" || "$CLEANUP_LOCAL_CKPT" == "true" || "$CLEANUP_LOCAL_CKPT" == "1" ]]; then
    echo ""
    echo "[4/4] Checking OSS backup before cleaning up local checkpoint ..."
    # Derive OSS path from experiment name and step
    oss_ckpt_dir="$CKPT_ROOT"
    if [[ -n "$STEP_TAG" ]]; then
        oss_experiment="$(basename "$CKPT_ROOT")"
        oss_step="$STEP_TAG"
    else
        oss_experiment="$(basename "$CKPT_ROOT")"
        oss_step=""
    fi
    oss_name="$(get_oss_name_from_exp_name "$oss_experiment")"
    if [[ -n "$oss_step" ]]; then
        oss_path="${OSS_BASE}/${oss_name}/${oss_step}"
    else
        oss_path="${OSS_BASE}/${oss_name}"
    fi

    if ossutil stat "${oss_path}/model.safetensors" > /dev/null 2>&1; then
        echo "  ✅ OSS backup verified at ${oss_path}"
        echo "  Removing local checkpoint directory: $CKPT_ROOT"
        safe_remove_checkpoint_dir "$CKPT_ROOT"
        # Remove parent experiment dir if empty
        local_parent="$(dirname "$CKPT_ROOT")"
        if [[ -d "$local_parent" ]] && [ -z "$(ls -A "$local_parent" 2>/dev/null)" ]; then
            rmdir "$local_parent" 2>/dev/null || true
        fi
        echo "  ✅ Local checkpoint cleaned up"
    else
        echo "  ⚠️  OSS backup NOT found at ${oss_path}/model.safetensors"
        echo "  Keeping local checkpoint to avoid data loss."
    fi
fi

echo ""
echo "Evaluation complete. Results saved to: $OUTPUT_DIR"
if [[ "$EVAL_FAILURES" -gt 0 ]]; then
    echo "Completed with ${EVAL_FAILURES} evaluation failure(s). See warnings above." >&2
    exit 1
fi

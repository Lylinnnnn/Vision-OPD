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
# Support both legacy (test.json) and full-scale (test_1000.json) datasets.
# Set DATASET_VERSION=full to use the full-scale dataset; default is full.
DATASET_VERSION="${DATASET_VERSION:-full}"
EVAL_MODE="${4:-${EVAL_MODE:-chair}}"
OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
CLEANUP_LOCAL_CKPT="${CLEANUP_LOCAL_CKPT:-True}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
TARGET_PX="${TARGET_PX:-448}"
PORT="${VLLM_PORT:-8000}"
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
POPE_PARALLEL_WORKERS="${POPE_PARALLEL_WORKERS:-64}"
POPE_MAX_NEW_TOKENS="${POPE_MAX_NEW_TOKENS:-16}"
POPE_MAX_SAMPLES="${POPE_MAX_SAMPLES:-0}"
POPE_USE_PREPARED_QUERY="${POPE_USE_PREPARED_QUERY:-False}"
CHAIR_MAX_NEW_TOKENS="${CHAIR_MAX_NEW_TOKENS:-384}"
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
VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-32768}"
VISION_PARALLEL_WORKERS="${VISION_PARALLEL_WORKERS:-128}"
VISION_MAX_RETRIES="${VISION_MAX_RETRIES:-3}"
VISION_ENABLE_THINKING="${VISION_ENABLE_THINKING:-}"
RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-False}"

normalize_eval_mode() {
    local mode
    mode="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    if [[ "$mode" == "frequent" ]]; then
        mode="chair,pope"
    fi
    if [[ "$mode" == "all" ]]; then
        mode="chair,pope,vision"
    fi
    echo "$mode"
}

has_eval_task() {
    local mode="$1"
    local task="$2"
    [[ ",${mode}," == *",${task},"* ]]
}

is_truthy() {
    [[ "$1" == "True" || "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "Y" || "$1" == "y" ]]
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
    if ! has_eval_task "$EVAL_MODE" "vision"; then
        return 0
    fi

    echo ""
    echo "[preflight] Preparing Vision-OPD benchmark data before vLLM ..."
    mkdir -p "$VISION_BENCHMARK_DATA_DIR"

    local old_ifs="$IFS"
    local benchmarks=()
    local bench
    IFS=',' read -r -a benchmarks <<< "$VISION_BENCHMARK"
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
        if is_truthy "$VISION_BENCHMARK_CLEAN_SOURCE"; then
            prepare_args+=(--clean-source)
        fi

        if [[ -s "$benchmark_json_path" ]]; then
            echo "  Found ${bench}: ${benchmark_json_path}"
            if is_truthy "$VISION_BENCHMARK_CLEAN_SOURCE"; then
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
if ! has_eval_task "$EVAL_MODE" "chair" && ! has_eval_task "$EVAL_MODE" "pope" && ! has_eval_task "$EVAL_MODE" "vision"; then
    echo "Error: eval_mode must include chair, pope, vision, frequent, or all. Got: $EVAL_MODE" >&2
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
    OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/full/${EXPERIMENT_NAME}/${DATASET_TAG}"
else
    OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${EXPERIMENT_NAME}/${DATASET_TAG}"
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
    echo "CHAIR logprobs: ${CHAIR_SAVE_LOGPROBS} (top=${CHAIR_TOP_LOGPROBS})"
    echo "OPD eval trace: ${EVAL_OPD_TRACE} (topk=${EVAL_OPD_TRACE_TOPK}, entropy=${EVAL_OPD_TRACE_ENTROPY})"
fi
if has_eval_task "$EVAL_MODE" "pope"; then
    echo "POPE:        $POPE_BENCHMARK"
    echo "POPE source: $POPE_SOURCE"
fi
if has_eval_task "$EVAL_MODE" "vision"; then
    echo "Vision-OPD:  $VISION_BENCHMARK"
    echo "Vision data: $VISION_BENCHMARK_DATA_DIR"
    echo "Vision auto download: $VISION_BENCHMARK_AUTO_DOWNLOAD"
    echo "Vision clean source:  $VISION_BENCHMARK_CLEAN_SOURCE"
fi
echo "Output:      $OUTPUT_DIR"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

prepare_vision_benchmark_data

# --- Step 1: Start vLLM server ---
echo ""
echo "[1/3] Starting vLLM server on port $PORT ..."
# Disable prometheus metrics to avoid '_IncludedRouter' compatibility issue with vLLM 0.18+
export VLLM_DISABLE_PROMETHEUS=1
"$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --gpu-memory-utilization 0.85 \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --port "$PORT" \
    --max-model-len 9728 \
    --disable-frontend-multiprocessing &
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
for i in $(seq 1 300); do
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
    echo "  vLLM server failed to start within 300s" >&2
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
    if ! "$PYTHON_BIN" "${RES_OPD_ROOT}/eval/eval_chair.py" \
        --api-base "http://localhost:$PORT/v1/" \
        --model-name "$MODEL_NAME" \
        --test-json "$TEST_JSON" \
        --output-dir "$OUTPUT_DIR" \
        --student-px "$STUDENT_PX" \
        --target-px "$TARGET_PX" \
        --degradation-mode "$DEGRADATION_MODE" \
        --student-ratio "$STUDENT_RATIO" \
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
        --parallel-workers "$POPE_PARALLEL_WORKERS" \
        ${pope_extra_args[@]+"${pope_extra_args[@]}"}; then
        echo "  WARNING: POPE failed; keeping any completed outputs." >&2
        EVAL_FAILURES=$((EVAL_FAILURES + 1))
    fi
fi

if has_eval_task "$EVAL_MODE" "vision"; then
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
            BENCHMARK="$VISION_BENCHMARK"
            BENCHMARK_OUTPUT_LAYOUT="per_benchmark"
            BENCHMARK_DATA_DIR="$VISION_BENCHMARK_DATA_DIR"
            BENCHMARK_PREPARE_DATA=False
            BENCHMARK_AUTO_DOWNLOAD="$VISION_BENCHMARK_AUTO_DOWNLOAD"
            BENCHMARK_CLEAN_SOURCE="$VISION_BENCHMARK_CLEAN_SOURCE"
            OUT_DIR="$vision_out_dir"
            JUDGE_DIR="$vision_judge_dir"
            MAX_TOKENS="$VISION_MAX_TOKENS"
            MAX_RETRIES="$VISION_MAX_RETRIES"
            PARALLEL_WORKERS="$VISION_PARALLEL_WORKERS"
            RULE_ONLY_JUDGE="$RULE_ONLY_JUDGE"
        )
        [[ -n "${OPENAI_API_KEY:-}" ]] && vision_args+=(OPENAI_API_KEY="$OPENAI_API_KEY")
        [[ -n "${JUDGE_API_BASE:-}" ]] && vision_args+=(JUDGE_API_BASE="$JUDGE_API_BASE")
        [[ -n "${JUDGE_API_KEY:-}" ]] && vision_args+=(JUDGE_API_KEY="$JUDGE_API_KEY")
        [[ -n "${JUDGE_MODEL:-}" ]] && vision_args+=(JUDGE_MODEL="$JUDGE_MODEL")
        [[ -n "${JUDGE_MODEL_PATH:-}" ]] && vision_args+=(JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH")
        [[ -n "${JUDGE_MAX_TOKENS:-}" ]] && vision_args+=(JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS")
        [[ -n "$VISION_ENABLE_THINKING" ]] && vision_args+=(ENABLE_THINKING="$VISION_ENABLE_THINKING")
        if ! env "${vision_args[@]}" bash "${VISION_OPD_ROOT}/eval/run_eval.sh"; then
            echo "  WARNING: Vision-OPD benchmark(s) failed; keeping any completed outputs." >&2
            EVAL_FAILURES=$((EVAL_FAILURES + 1))
        fi
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
        oss_experiment="$(basename "$(dirname "$CKPT_ROOT")")"
        oss_step="$STEP_TAG"
    else
        oss_experiment="$(basename "$CKPT_ROOT")"
        oss_step=""
    fi
    # Convert local naming (Res-OPD-Qwen3VL-2B-Instruct-...) to OSS naming (ResOPD_...)
    oss_name=$(echo "$oss_experiment" | sed \
        -e 's/^Res-OPD-Qwen3VL-2B-Instruct-/ResOPD_/' \
        -e 's/-orig-sr/_orig_sr/' \
        -e 's/-tr/_tr/' \
        -e 's/-a/_a/' \
        -e 's/-ema/_ema/' \
        -e 's/-frozen/_frozen/' \
        -e 's/-rkl/_rkl/' \
        -e 's/-veto/_veto/' \
        -e 's/-s/_s/' \
        -e 's/-t/_t/' \
        -e 's/-e/_e/' \
        -e 's/\./_/g')
    if [[ -n "$oss_step" ]]; then
        oss_path="${OSS_BASE}/${oss_name}/${oss_step}"
    else
        oss_path="${OSS_BASE}/${oss_name}"
    fi

    if ossutil stat "${oss_path}/model.safetensors" > /dev/null 2>&1; then
        echo "  ✅ OSS backup verified at ${oss_path}"
        echo "  Removing local checkpoint directory: $CKPT_ROOT"
        rm -rf "$CKPT_ROOT"
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

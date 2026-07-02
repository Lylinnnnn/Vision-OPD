#!/usr/bin/env bash

set -euo pipefail

# Deterministic data-sharded eval wrapper for eval_after_merge.sh.
# It launches one vLLM server per GPU with tensor parallel size 1, evaluates
# modulo shards, merges JSONL outputs, recomputes metrics once, then removes
# temporary shard files on success.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"

MODEL_PATH="${1:?Usage: $0 <merged_checkpoint_path> [student_px] [version_tag] [eval_mode]}"
STUDENT_PX="${2:-${STUDENT_PX:-}}"
VERSION_TAG="${3:-latest}"
EVAL_MODE="${4:-${EVAL_MODE:-chair}}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
EVAL_SCRIPT="${RES_OPD_ROOT}/scripts/eval_after_merge.sh"
MERGE_SCRIPT="${RES_OPD_ROOT}/eval/merge_sharded_eval.py"

EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"
SHARDED_EVAL_KEEP_SHARDS="${SHARDED_EVAL_KEEP_SHARDS:-False}"
SHARDED_EVAL_RUN_ID="${SHARDED_EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
SHARDED_EVAL_SESSION_PREFIX="${SHARDED_EVAL_SESSION_PREFIX:-resopd_eval_${SHARDED_EVAL_RUN_ID}}"
SHARDED_EVAL_FORCE_TP1="${SHARDED_EVAL_FORCE_TP1:-True}"
SHARDED_EVAL_POLL_SECONDS="${SHARDED_EVAL_POLL_SECONDS:-10}"
SHARDED_EVAL_ALLOW_OPD_TRACE="${SHARDED_EVAL_ALLOW_OPD_TRACE:-False}"

DATASET_VERSION="${DATASET_VERSION:-full}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
TEACHER_RATIO="${TEACHER_RATIO:-}"
TEACHER_PX="${TEACHER_PX:-}"
TARGET_PX="${TARGET_PX:-448}"
RESULT_VERSION_TAG="${RESULT_VERSION_TAG:-}"
EVAL_OPD_TRACE="${EVAL_OPD_TRACE:-False}"
VISION_BENCHMARK="${VISION_BENCHMARK:-mmstar}"
VISION_BENCHMARK_DATA_DIR="${VISION_BENCHMARK_DATA_DIR:-${BENCHMARK_DATA_DIR:-/home/liuyanlin.lyl/notebook/data}}"
VISION_BENCHMARK_OUTPUT_SUFFIX="${VISION_BENCHMARK_OUTPUT_SUFFIX:-${BENCHMARK_OUTPUT_SUFFIX:-}}"
RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-}"
MCQ_EXTRACT_MODE="${MCQ_EXTRACT_MODE:-}"
POPE_BENCHMARK="${POPE_BENCHMARK:-pope_adv,pope_pop,pope_random}"
POPE_SOURCE="${POPE_SOURCE:-res-opd-test}"
SEED="${SEED:-42}"
AMBER_ROOT="${AMBER_ROOT:-${BENCHMARK_DATA_DIR:-/home/liuyanlin.lyl/notebook/data}/AMBER}"
AMBER_EVAL_TYPE="${AMBER_EVAL_TYPE:-a}"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"

is_truthy() {
    [[ "${1:-}" == "True" || "${1:-}" == "true" || "${1:-}" == "1" || "${1:-}" == "yes" || "${1:-}" == "Y" || "${1:-}" == "y" ]]
}

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
                normalized+=(chair pope vision)
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
                echo "Error: unsupported sharded eval task '$token'. Use chair,pope,mmstar,cv-bench,vision,amber,frequent,all." >&2
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
    local explicit_profile="${MODEL_PROFILE:-}"
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
            echo "$VERSION_TAG"
            ;;
    esac
}

infer_eval_spec() {
    local exp_name="$1"
    local inferred_mode="square"
    local inferred_student_px="0"
    local inferred_teacher_px="$TARGET_PX"
    local inferred_student_ratio="1.0"
    local inferred_teacher_ratio="1.0"

    case "$(infer_model_profile)" in
        qwen3vl_2b_thinking|qwen3vl_8b_thinking|generic_thinking)
            inferred_mode="original"
            inferred_student_px="0"
            inferred_student_ratio="1.0"
            ;;
    esac

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
        *) echo "" ;;
    esac
}

prepare_vision_benchmark_data_once() {
    if ! has_vision_eval_task "$EVAL_MODE"; then
        return 0
    fi
    mkdir -p "$VISION_BENCHMARK_DATA_DIR"
    local old_ifs="$IFS"
    local benchmarks=()
    local bench
    IFS=',' read -r -a benchmarks <<< "$EFFECTIVE_VISION_BENCHMARK"
    IFS="$old_ifs"

    echo "[preflight] Preparing Vision-OPD benchmark data once before shard launch ..."
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
        if [[ -s "$benchmark_json_path" ]]; then
            echo "  Found ${bench}: ${benchmark_json_path}"
            continue
        fi
        if ! is_truthy "${VISION_BENCHMARK_AUTO_DOWNLOAD:-${BENCHMARK_AUTO_DOWNLOAD:-True}}"; then
            echo "Error: benchmark JSON missing and auto-download is disabled: ${benchmark_json_path}" >&2
            exit 1
        fi
        local -a prepare_args=(--benchmark "$bench" --data_dir "$VISION_BENCHMARK_DATA_DIR")
        if is_truthy "${VISION_BENCHMARK_REFRESH_PROMPTS:-${BENCHMARK_REFRESH_PROMPTS:-False}}"; then
            prepare_args+=(--refresh-prompts)
        fi
        if is_truthy "${VISION_BENCHMARK_CLEAN_SOURCE:-${BENCHMARK_CLEAN_SOURCE:-False}}"; then
            prepare_args+=(--clean-source)
        fi
        "$PYTHON_BIN" "${VISION_OPD_ROOT}/eval/prepare_data.py" "${prepare_args[@]}"
        if [[ ! -s "$benchmark_json_path" ]]; then
            echo "Error: prepared benchmark JSON is missing or empty: ${benchmark_json_path}" >&2
            exit 1
        fi
    done
}

write_export() {
    local file="$1"
    local name="$2"
    local value="$3"
    printf 'export %s=%q\n' "$name" "$value" >> "$file"
}

require_tmux() {
    if ! command -v tmux >/dev/null 2>&1; then
        echo "Error: tmux is required for sharded eval." >&2
        exit 1
    fi
}

EVAL_MODE="$(normalize_eval_mode "$EVAL_MODE")"
EFFECTIVE_VISION_BENCHMARK="$(resolve_vision_benchmarks "$EVAL_MODE")"
MODEL_PROFILE="$(infer_model_profile)"
RESULT_VERSION_TAG="$(resolve_result_version_tag "$MODEL_PROFILE")"

if has_vision_eval_task "$EVAL_MODE"; then
    if [[ -z "$RULE_ONLY_JUDGE" ]]; then
        RULE_ONLY_JUDGE="True"
    fi
    if [[ -z "$MCQ_EXTRACT_MODE" ]]; then
        MCQ_EXTRACT_MODE="official"
    fi
fi

CKPT_ROOT="$MODEL_PATH"
STEP_TAG=""
if [[ "$(basename "$MODEL_PATH")" =~ ^global_step_[0-9]+$ ]]; then
    STEP_TAG="$(basename "$MODEL_PATH")"
    CKPT_ROOT="$(dirname "$MODEL_PATH")"
fi
EXPERIMENT_NAME="$(basename "$CKPT_ROOT")"
if [[ -n "$STEP_TAG" ]]; then
    EXPERIMENT_NAME="${EXPERIMENT_NAME}_${STEP_TAG}"
fi
infer_eval_spec "$EXPERIMENT_NAME"

case "$DEGRADATION_MODE" in
    square|original)
        ;;
    *)
        echo "Error: DEGRADATION_MODE must be square or original. Got: $DEGRADATION_MODE" >&2
        exit 1
        ;;
esac

if [[ "$DATASET_VERSION" == "full" ]]; then
    TRAIN_FILE="${RES_OPD_ROOT}/data/train_5k.parquet"
    TEST_FILE="${RES_OPD_ROOT}/data/test_1000.json"
else
    TRAIN_FILE="${RES_OPD_ROOT}/data/train.parquet"
    TEST_FILE="${RES_OPD_ROOT}/data/test.json"
fi
if [[ -f "$TRAIN_FILE" && -f "$TEST_FILE" ]]; then
    TRAIN_N=$("$PYTHON_BIN" -c "import pandas as pd; print(len(pd.read_parquet('$TRAIN_FILE')))" 2>/dev/null || echo "?")
    TEST_N=$("$PYTHON_BIN" -c "import json; print(len(json.load(open('$TEST_FILE'))))" 2>/dev/null || echo "?")
    DATASET_TAG="train${TRAIN_N}_test${TEST_N}"
else
    DATASET_TAG="unknown_dataset"
fi
if [[ "$DEGRADATION_MODE" == "original" ]]; then
    RATIO_TAG="${STUDENT_RATIO//./p}"
    DATASET_TAG="${DATASET_TAG}_original_sr${RATIO_TAG}"
fi

if [[ -n "${EVAL_OUTPUT_DIR:-}" ]]; then
    FINAL_OUTPUT_DIR="$EVAL_OUTPUT_DIR"
elif [[ "$DATASET_VERSION" == "full" ]]; then
    FINAL_OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${RESULT_VERSION_TAG}/full/${EXPERIMENT_NAME}/${DATASET_TAG}"
else
    FINAL_OUTPUT_DIR="${RES_OPD_ROOT}/eval_results/${RESULT_VERSION_TAG}/${EXPERIMENT_NAME}/${DATASET_TAG}"
fi

SHARD_ROOT="${SHARDED_EVAL_WORK_DIR:-${FINAL_OUTPUT_DIR}/.shards/${SHARDED_EVAL_RUN_ID}}"

if ! [[ "$EVAL_SHARD_COUNT" =~ ^[0-9]+$ ]] || [[ "$EVAL_SHARD_COUNT" -le 0 ]]; then
    echo "Error: EVAL_SHARD_COUNT must be a positive integer. Got: ${EVAL_SHARD_COUNT}" >&2
    exit 1
fi

IFS=',' read -r -a GPUS <<< "$GPU_LIST"
if [[ "${#GPUS[@]}" -lt "$EVAL_SHARD_COUNT" ]]; then
    echo "Error: GPU_LIST has ${#GPUS[@]} entries but EVAL_SHARD_COUNT=${EVAL_SHARD_COUNT}: ${GPU_LIST}" >&2
    exit 1
fi

require_tmux
prepare_vision_benchmark_data_once

mkdir -p "$SHARD_ROOT" "$FINAL_OUTPUT_DIR"
ENV_DUMP="${SHARD_ROOT}/parent_env.sh"
export -p > "$ENV_DUMP"

if is_truthy "$EVAL_OPD_TRACE" && ! is_truthy "$SHARDED_EVAL_ALLOW_OPD_TRACE"; then
    echo "WARNING: disabling EVAL_OPD_TRACE inside sharded eval; shard trace merge is not supported." >&2
fi

echo "============================================================"
echo " Res-OPD Sharded Evaluation"
echo "============================================================"
echo "Model:       $MODEL_PATH"
echo "Eval mode:   $EVAL_MODE"
echo "Vision:      ${EFFECTIVE_VISION_BENCHMARK:-<none>}"
echo "Result tag:  $RESULT_VERSION_TAG"
echo "Output:      $FINAL_OUTPUT_DIR"
echo "Shard root:  $SHARD_ROOT"
echo "Shard count: $EVAL_SHARD_COUNT"
echo "GPU list:    $GPU_LIST"
echo "Base port:   $VLLM_BASE_PORT"
echo "TP per shard: $(is_truthy "$SHARDED_EVAL_FORCE_TP1" && echo 1 || echo '<env/default>')"
echo "============================================================"

sessions=()
for shard_idx in $(seq 0 $((EVAL_SHARD_COUNT - 1))); do
    gpu="${GPUS[$shard_idx]}"
    shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "$shard_idx")"
    runner="${shard_dir}/run.sh"
    log_file="${shard_dir}/eval.log"
    exit_file="${shard_dir}/exit_code"
    session="${SHARDED_EVAL_SESSION_PREFIX}_$(printf '%02d' "$shard_idx")"
    mkdir -p "$shard_dir"

    {
        echo "#!/usr/bin/env bash"
        echo "set -uo pipefail"
        printf 'source %q\n' "$ENV_DUMP"
        write_export /dev/stdout "CUDA_VISIBLE_DEVICES" "$gpu"
        write_export /dev/stdout "VLLM_PORT" "$((VLLM_BASE_PORT + shard_idx))"
        write_export /dev/stdout "EVAL_SHARD_COUNT" "$EVAL_SHARD_COUNT"
        write_export /dev/stdout "EVAL_SHARD_INDEX" "$shard_idx"
        write_export /dev/stdout "EVAL_OUTPUT_DIR" "${shard_dir}/out"
        write_export /dev/stdout "EVAL_SHARDED_CHILD" "1"
        write_export /dev/stdout "CLEANUP_LOCAL_CKPT" "False"
        write_export /dev/stdout "BENCHMARK_SKIP_JUDGE" "True"
        write_export /dev/stdout "AMBER_SKIP_OFFICIAL_EVAL" "True"
        write_export /dev/stdout "RESULT_VERSION_TAG" "$RESULT_VERSION_TAG"
        write_export /dev/stdout "RULE_ONLY_JUDGE" "$RULE_ONLY_JUDGE"
        write_export /dev/stdout "MCQ_EXTRACT_MODE" "$MCQ_EXTRACT_MODE"
        write_export /dev/stdout "VISION_BENCHMARK" "$EFFECTIVE_VISION_BENCHMARK"
        write_export /dev/stdout "VISION_BENCHMARK_AUTO_DOWNLOAD" "False"
        write_export /dev/stdout "VISION_BENCHMARK_CLEAN_SOURCE" "False"
        write_export /dev/stdout "BENCHMARK_AUTO_DOWNLOAD" "False"
        write_export /dev/stdout "BENCHMARK_CLEAN_SOURCE" "False"
        write_export /dev/stdout "EVAL_BACKEND" "single"
        if is_truthy "$SHARDED_EVAL_FORCE_TP1"; then
            write_export /dev/stdout "VLLM_TENSOR_PARALLEL_SIZE" "1"
        fi
        if is_truthy "$EVAL_OPD_TRACE" && ! is_truthy "$SHARDED_EVAL_ALLOW_OPD_TRACE"; then
            write_export /dev/stdout "EVAL_OPD_TRACE" "False"
        fi
        printf 'cd %q\n' "$VISION_OPD_ROOT"
        printf 'echo "[shard %s/%s] gpu=%s port=%s" > %q\n' "$shard_idx" "$EVAL_SHARD_COUNT" "$gpu" "$((VLLM_BASE_PORT + shard_idx))" "$log_file"
        printf 'bash %q %q %q %q %q >> %q 2>&1\n' "$EVAL_SCRIPT" "$MODEL_PATH" "$STUDENT_PX" "$VERSION_TAG" "$EVAL_MODE" "$log_file"
        echo 'status=$?'
        printf 'echo "$status" > %q\n' "$exit_file"
        echo 'exit "$status"'
    } > "$runner"
    chmod +x "$runner"

    if tmux has-session -t "$session" 2>/dev/null; then
        tmux kill-session -t "$session"
    fi
    tmux new-session -d -s "$session" "bash '$runner'"
    sessions+=("$session")
    echo "Started shard ${shard_idx}/${EVAL_SHARD_COUNT} on GPU ${gpu}, port $((VLLM_BASE_PORT + shard_idx)), tmux=${session}"
done

cleanup_sessions() {
    local session
    for session in "${sessions[@]:-}"; do
        tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true
    done
}
trap cleanup_sessions INT TERM

echo "Waiting for shard tmux sessions ..."
while true; do
    done_count=0
    failure_count=0
    for shard_idx in $(seq 0 $((EVAL_SHARD_COUNT - 1))); do
        shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "$shard_idx")"
        exit_file="${shard_dir}/exit_code"
        session="${SHARDED_EVAL_SESSION_PREFIX}_$(printf '%02d' "$shard_idx")"
        if [[ -f "$exit_file" ]]; then
            done_count=$((done_count + 1))
            if [[ "$(cat "$exit_file")" != "0" ]]; then
                failure_count=$((failure_count + 1))
            fi
        elif ! tmux has-session -t "$session" 2>/dev/null; then
            echo "255" > "$exit_file"
            done_count=$((done_count + 1))
            failure_count=$((failure_count + 1))
        fi
    done
    echo "  shards done: ${done_count}/${EVAL_SHARD_COUNT}"
    if [[ "$done_count" -ge "$EVAL_SHARD_COUNT" ]]; then
        break
    fi
    sleep "$SHARDED_EVAL_POLL_SECONDS"
done
trap - INT TERM

failed=0
for shard_idx in $(seq 0 $((EVAL_SHARD_COUNT - 1))); do
    shard_dir="${SHARD_ROOT}/shard_$(printf '%02d' "$shard_idx")"
    status="$(cat "${shard_dir}/exit_code")"
    if [[ "$status" != "0" ]]; then
        failed=1
        echo "---- shard ${shard_idx} failed with status ${status}; tail log ----" >&2
        tail -n 120 "${shard_dir}/eval.log" >&2 || true
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "Error: one or more shards failed. Temporary shard outputs kept at: ${SHARD_ROOT}" >&2
    exit 1
fi

echo "[merge] Recomputing metrics from shards ..."
merge_args=(
    --final-output-dir "$FINAL_OUTPUT_DIR"
    --shard-root "$SHARD_ROOT"
    --eval-mode "$EVAL_MODE"
    --pope-benchmark "$POPE_BENCHMARK"
    --pope-source "$POPE_SOURCE"
    --vision-benchmark "$EFFECTIVE_VISION_BENCHMARK"
    --vision-benchmark-data-dir "$VISION_BENCHMARK_DATA_DIR"
    --vision-benchmark-output-suffix "$VISION_BENCHMARK_OUTPUT_SUFFIX"
    --experiment-name "$EXPERIMENT_NAME"
    --seed "$SEED"
    --mcq-extract-mode "$MCQ_EXTRACT_MODE"
    --judge-max-tokens "${JUDGE_MAX_TOKENS:-2048}"
    --amber-root "$AMBER_ROOT"
    --amber-eval-type "$AMBER_EVAL_TYPE"
)
is_truthy "$RULE_ONLY_JUDGE" && merge_args+=(--rule-only-judge)
[[ -n "${JUDGE_API_BASE:-}" ]] && merge_args+=(--judge-api-base "$JUDGE_API_BASE")
[[ -n "${JUDGE_API_KEY:-}" ]] && merge_args+=(--judge-api-key "$JUDGE_API_KEY")
[[ -n "${JUDGE_MODEL:-}" ]] && merge_args+=(--judge-model "$JUDGE_MODEL")
[[ -n "${JUDGE_MODEL_PATH:-}" ]] && merge_args+=(--judge-model-path "$JUDGE_MODEL_PATH")
if is_truthy "${AMBER_SKIP_OFFICIAL_EVAL:-False}"; then
    merge_args+=(--amber-skip-official-eval)
fi
[[ -n "${AMBER_WORD_ASSOCIATION:-}" ]] && merge_args+=(--amber-word-association "$AMBER_WORD_ASSOCIATION")
[[ -n "${AMBER_SAFE_WORDS:-}" ]] && merge_args+=(--amber-safe-words "$AMBER_SAFE_WORDS")
[[ -n "${AMBER_ANNOTATION:-}" ]] && merge_args+=(--amber-annotation "$AMBER_ANNOTATION")
[[ -n "${AMBER_METRICS:-}" ]] && merge_args+=(--amber-metrics "$AMBER_METRICS")

"$PYTHON_BIN" "$MERGE_SCRIPT" "${merge_args[@]}"

if ! is_truthy "$SHARDED_EVAL_KEEP_SHARDS"; then
    rm -rf "$SHARD_ROOT"
    parent_shards="$(dirname "$SHARD_ROOT")"
    rmdir "$parent_shards" 2>/dev/null || true
else
    echo "Keeping shard outputs at: $SHARD_ROOT"
fi

echo "Sharded evaluation complete. Results saved to: $FINAL_OUTPUT_DIR"

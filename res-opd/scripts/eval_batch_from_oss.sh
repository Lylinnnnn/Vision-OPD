#!/bin/bash
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ⚠️  Log 规则: 批量 eval 无单独 log 文件，结果存储在
#     eval_results/<version_tag>/<experiment_name>_<step>/<dataset>/
#     禁止在 logs/ 下创建 eval 相关子目录！
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Batch Evaluation from OSS (Parameterized)
# Downloads merged checkpoints from OSS, runs eval, deletes model files,
# keeps only eval results.
#
# Usage:
#   bash scripts/eval_batch_from_oss.sh --experiment-names <exp1> [exp2] ... \
#       [--step global_step_92] [--student-ratio 1.0] [--degradation-mode original] \
#       [--student-ratio 1.0] [--version-tag v5] [--eval-mode chair,pope] \
#       [--vision-benchmark mmstar,cv-bench] [--vision-benchmark-data-dir /home/liuyanlin.lyl/notebook/data] \
#       [--chair-save-logprobs true] [--chair-top-logprobs 5]
#
#   bash scripts/eval_batch_from_oss.sh --oss-names <name1> [name2] ... \
#       [--step global_step_92] [--student-ratio 1.0] [--degradation-mode original] \
#       [--student-ratio 1.0] [--version-tag v5] [--eval-mode chair,pope] \
#       [--vision-benchmark mmstar,cv-bench] [--vision-benchmark-data-dir /home/liuyanlin.lyl/notebook/data] \
#       [--chair-save-logprobs true] [--chair-top-logprobs 5]
#
# Step behavior:
#   - Omit --step to evaluate every OSS global_step_* checkpoint serially.
#   - Pass --step global_step_50 (or --step 50) to evaluate one checkpoint.
#
# Experiment name behavior:
#   - Pass --experiment-names / --oss-names for explicit paths.
#   - Omit both to construct one experiment name from current protocol params
#     such as MODEL_PATH, STUDENT_RATIO, TEACHER_RATIO, TEACHER_MODE, ALPHA,
#     TRAIN_BATCH_SIZE, ROLLOUT_N, OPD_SELECTIVE_WEIGHT, and DATASET_VERSION.
#
# eval-mode:
#   chair,pope       Default local COCO eval.
#   chair / pope     Run one local COCO eval.
#   vision           Run Vision-OPD auxiliary eval; controlled by --vision-benchmark.
#   amber / mme      Optional final hallucination benchmarks; MME only runs when explicitly requested.
#   all              Run chair,pope,vision,amber.
#
# Examples:
#   # Evaluate by local experiment names; OSS names are derived with the same
#   # mapping used by run_res_opd.sh / ckpt_upload_watcher.sh.
#   bash scripts/eval_batch_from_oss.sh \
#       --experiment-names Res-OPD-Qwen3-VL-2B-Thinking-orig-sr1.0-tr0.75-a1.0-frozen-rkl-b8-rn4-full5k-e1 \
#       --step global_step_39
#
#   # Evaluate specific legacy OSS experiments at step 92
#   bash scripts/eval_batch_from_oss.sh \
#       --oss-names ResOPD_s448_t0_a0.5_ema-e2 ResOPD_s448_t448_a0.5_ema-e2 \
#       --step global_step_92 --student-px 448 --version-tag v5
#
#   # Evaluate all ema experiments at default step
#   bash scripts/eval_batch_from_oss.sh \
#       --oss-names ResOPD_s448_t200_a0.5_ema-e2 \
#       --step global_step_92
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
source "${SCRIPT_DIR}/path_utils.sh"
cd "$VISION_OPD_ROOT"

OSS_BASE="oss://industry-algo/yanlin/ckpt/OPD/v4"
CKPT_BASE="${RES_OPD_ROOT}/checkpoints"
EVAL_SCRIPT="${RES_OPD_ROOT}/scripts/eval_after_merge.sh"
SHARDED_EVAL_SCRIPT="${RES_OPD_ROOT}/scripts/eval_after_merge_sharded_8gpu.sh"
AMBER_SCRIPT="${RES_OPD_ROOT}/scripts/archive/val_amber.sh"
MME_SCRIPT="${RES_OPD_ROOT}/scripts/archive/val_mme_perception.sh"
VISION_STATUS_SCRIPT="${RES_OPD_ROOT}/eval/check_vision_opd_eval.py"
HF_FILES=(
    config.json
    tokenizer_config.json
    tokenizer.json
    chat_template.jinja
    generation_config.json
    processor_config.json
    preprocessor_config.json
    image_processor_config.json
    video_processor_config.json
    special_tokens_map.json
    tokenizer.model
    merges.txt
    vocab.json
    model.safetensors.index.json
)

# Defaults
STEP="${STEP:-}"
STUDENT_PX="${STUDENT_PX:-}"
TEACHER_PX="${TEACHER_PX:-}"
TARGET_PX="${TARGET_PX:-448}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
TEACHER_RATIO="${TEACHER_RATIO:-}"
VERSION_TAG="latest"
RESULT_VERSION_TAG="${RESULT_VERSION_TAG:-}"
DATASET_VERSION="${DATASET_VERSION:-full}"
EVAL_MODE="${EVAL_MODE:-chair,pope}"
EVAL_BACKEND="${EVAL_BACKEND:-sharded}"
EVAL_SHARDED="${EVAL_SHARDED:-False}"
EVAL_SHARD_COUNT="${EVAL_SHARD_COUNT:-8}"
GPU_LIST="${GPU_LIST:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
SHARDED_EVAL_KEEP_SHARDS="${SHARDED_EVAL_KEEP_SHARDS:-False}"
DEFAULT_MODEL_ROOT="$(res_opd_default_model_root)"
DEFAULT_BENCHMARK_DATA_DIR="$(res_opd_default_data_root)"
PYTHON_BIN="${PYTHON_BIN:-$(res_opd_default_python_bin)}"
MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_ROOT}/Qwen3VL-2B-Instruct}"
MODEL_NAME="${MODEL_NAME:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-}"
TEACHER_MODE="${TEACHER_MODE:-}"
ALPHA="${ALPHA:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
ROLLOUT_N="${ROLLOUT_N:-}"
OPD_SELECTIVE_WEIGHT="${OPD_SELECTIVE_WEIGHT:-}"
OPD_SELECTIVE_WEIGHT_MODE="${OPD_SELECTIVE_WEIGHT_MODE:-}"
OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE="${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE:-}"
OPD_RISK_MASK_TOP_P="${OPD_RISK_MASK_TOP_P:-}"
OPD_TOKEN_MASK_PCT="${OPD_TOKEN_MASK_PCT:-}"
VLLM_PORT_CLEANUP="${VLLM_PORT_CLEANUP:-True}"
VLLM_PORT_CLEANUP_WAIT="${VLLM_PORT_CLEANUP_WAIT:-20}"
EVAL_MAX_TOKENS="${EVAL_MAX_TOKENS:-8192}"
EVAL_THINKING_MAX_TOKENS="${EVAL_THINKING_MAX_TOKENS:-$EVAL_MAX_TOKENS}"
CHAIR_MAX_NEW_TOKENS_WAS_SET="${CHAIR_MAX_NEW_TOKENS+x}"
CHAIR_MAX_NEW_TOKENS="${CHAIR_MAX_NEW_TOKENS:-$EVAL_MAX_TOKENS}"
CHAIR_PARALLEL_WORKERS="${CHAIR_PARALLEL_WORKERS:-8}"
CHAIR_MAX_SAMPLES="${CHAIR_MAX_SAMPLES:-0}"
CHAIR_SAVE_LOGPROBS="${CHAIR_SAVE_LOGPROBS:-False}"
CHAIR_TOP_LOGPROBS="${CHAIR_TOP_LOGPROBS:-5}"
POPE_MAX_NEW_TOKENS_WAS_SET="${POPE_MAX_NEW_TOKENS+x}"
POPE_MAX_NEW_TOKENS="${POPE_MAX_NEW_TOKENS:-$EVAL_MAX_TOKENS}"
POPE_PARALLEL_WORKERS="${POPE_PARALLEL_WORKERS:-64}"
POPE_MAX_SAMPLES="${POPE_MAX_SAMPLES:-0}"
POPE_USE_PREPARED_QUERY="${POPE_USE_PREPARED_QUERY:-False}"
EVAL_OPD_TRACE="${EVAL_OPD_TRACE:-False}"
EVAL_OPD_TRACE_TOPK="${EVAL_OPD_TRACE_TOPK:-50}"
EVAL_OPD_TRACE_ENTROPY="${EVAL_OPD_TRACE_ENTROPY:-True}"
EVAL_OPD_TRACE_SCORE_BASELINE="${EVAL_OPD_TRACE_SCORE_BASELINE:-False}"
EVAL_OPD_TRACE_CASE_ANALYSIS="${EVAL_OPD_TRACE_CASE_ANALYSIS:-}"
EVAL_OPD_TRACE_MAX_SAMPLES="${EVAL_OPD_TRACE_MAX_SAMPLES:-0}"
VISION_BENCHMARK="${VISION_BENCHMARK:-mmstar,cv-bench}"
VISION_BENCHMARK_DATA_DIR="${VISION_BENCHMARK_DATA_DIR:-${BENCHMARK_DATA_DIR:-$DEFAULT_BENCHMARK_DATA_DIR}}"
VISION_BENCHMARK_AUTO_DOWNLOAD="${VISION_BENCHMARK_AUTO_DOWNLOAD:-${BENCHMARK_AUTO_DOWNLOAD:-True}}"
VISION_BENCHMARK_CLEAN_SOURCE="${VISION_BENCHMARK_CLEAN_SOURCE:-${BENCHMARK_CLEAN_SOURCE:-False}}"
VISION_BENCHMARK_REFRESH_PROMPTS_WAS_SET="${VISION_BENCHMARK_REFRESH_PROMPTS+x}${BENCHMARK_REFRESH_PROMPTS+x}"
VISION_BENCHMARK_REFRESH_PROMPTS="${VISION_BENCHMARK_REFRESH_PROMPTS:-${BENCHMARK_REFRESH_PROMPTS:-False}}"
VISION_BENCHMARK_OUTPUT_SUFFIX="${VISION_BENCHMARK_OUTPUT_SUFFIX:-${BENCHMARK_OUTPUT_SUFFIX:-}}"
VISION_MAX_TOKENS_WAS_SET="${VISION_MAX_TOKENS+x}"
VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-$EVAL_MAX_TOKENS}"
VISION_PARALLEL_WORKERS="${VISION_PARALLEL_WORKERS:-128}"
VISION_MAX_RETRIES="${VISION_MAX_RETRIES:-3}"
VISION_ENABLE_THINKING="${VISION_ENABLE_THINKING:-}"
RULE_ONLY_JUDGE_WAS_SET="${RULE_ONLY_JUDGE+x}"
RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-True}"
MCQ_EXTRACT_MODE_WAS_SET="${MCQ_EXTRACT_MODE+x}"
MCQ_EXTRACT_MODE="${MCQ_EXTRACT_MODE:-legacy}"
FORCE_VISION_EVAL="${FORCE_VISION_EVAL:-False}"
JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
AMBER_ROOT="${AMBER_ROOT:-${BENCHMARK_DATA_DIR:-$DEFAULT_BENCHMARK_DATA_DIR}/AMBER}"
AMBER_IMAGE_ROOT="${AMBER_IMAGE_ROOT:-}"
AMBER_EVAL_TYPE="${AMBER_EVAL_TYPE:-a}"
AMBER_MAX_SAMPLES="${AMBER_MAX_SAMPLES:-0}"
AMBER_PARALLEL_WORKERS="${AMBER_PARALLEL_WORKERS:-64}"
AMBER_OFFICIAL_EVAL_WORKERS="${AMBER_OFFICIAL_EVAL_WORKERS:-16}"
AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET="${AMBER_MAX_NEW_TOKENS_GENERATIVE+x}"
AMBER_MAX_NEW_TOKENS_GENERATIVE="${AMBER_MAX_NEW_TOKENS_GENERATIVE:-$EVAL_MAX_TOKENS}"
AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET="${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE+x}"
AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE:-$EVAL_MAX_TOKENS}"
AMBER_SKIP_OFFICIAL_EVAL="${AMBER_SKIP_OFFICIAL_EVAL:-False}"
AMBER_WORD_ASSOCIATION="${AMBER_WORD_ASSOCIATION:-}"
AMBER_SAFE_WORDS="${AMBER_SAFE_WORDS:-}"
AMBER_ANNOTATION="${AMBER_ANNOTATION:-}"
AMBER_METRICS="${AMBER_METRICS:-}"
OSS_NAMES=()
EXPERIMENT_NAMES=()
LOCAL_NAMES=()
STUDENT_RATIOS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --oss-names)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OSS_NAMES+=("$1")
                shift
            done
            ;;
        --experiment-names)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                EXPERIMENT_NAMES+=("$1")
                shift
            done
            ;;
        --local-names)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LOCAL_NAMES+=("$1")
                shift
            done
            ;;
        --student-ratios)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                STUDENT_RATIOS+=("$1")
                shift
            done
            ;;
        --step)       STEP="$2"; shift 2 ;;
        --student-px) STUDENT_PX="$2"; shift 2 ;;
        --teacher-px) TEACHER_PX="$2"; shift 2 ;;
        --target-px) TARGET_PX="$2"; shift 2 ;;
        --degradation-mode) DEGRADATION_MODE="$2"; shift 2 ;;
        --student-ratio) STUDENT_RATIO="$2"; shift 2 ;;
        --teacher-ratio) TEACHER_RATIO="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --model-name) MODEL_NAME="$2"; shift 2 ;;
        --total-epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
        --teacher-mode) TEACHER_MODE="$2"; shift 2 ;;
        --alpha) ALPHA="$2"; shift 2 ;;
        --train-batch-size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
        --rollout-n) ROLLOUT_N="$2"; shift 2 ;;
        --opd-selective-weight) OPD_SELECTIVE_WEIGHT="$2"; shift 2 ;;
        --opd-selective-weight-mode) OPD_SELECTIVE_WEIGHT_MODE="$2"; shift 2 ;;
        --opd-selective-weight-uncertainty-mode) OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE="$2"; shift 2 ;;
        --opd-risk-mask-top-p) OPD_RISK_MASK_TOP_P="$2"; shift 2 ;;
        --opd-token-mask-pct) OPD_TOKEN_MASK_PCT="$2"; shift 2 ;;
        --version-tag) VERSION_TAG="$2"; shift 2 ;;
        --result-version-tag) RESULT_VERSION_TAG="$2"; shift 2 ;;
        --eval-mode)  EVAL_MODE="$2"; shift 2 ;;
        --eval-backend) EVAL_BACKEND="$2"; shift 2 ;;
        --eval-sharded|--sharded-eval) EVAL_SHARDED="$2"; shift 2 ;;
        --eval-shard-count) EVAL_SHARD_COUNT="$2"; shift 2 ;;
        --gpu-list) GPU_LIST="$2"; shift 2 ;;
        --sharded-eval-keep-shards) SHARDED_EVAL_KEEP_SHARDS="$2"; shift 2 ;;
        --chair-max-new-tokens) CHAIR_MAX_NEW_TOKENS="$2"; CHAIR_MAX_NEW_TOKENS_WAS_SET=arg; shift 2 ;;
        --chair-parallel-workers) CHAIR_PARALLEL_WORKERS="$2"; shift 2 ;;
        --chair-max-samples) CHAIR_MAX_SAMPLES="$2"; shift 2 ;;
        --chair-save-logprobs) CHAIR_SAVE_LOGPROBS="$2"; shift 2 ;;
        --chair-top-logprobs) CHAIR_TOP_LOGPROBS="$2"; shift 2 ;;
        --pope-max-new-tokens) POPE_MAX_NEW_TOKENS="$2"; POPE_MAX_NEW_TOKENS_WAS_SET=arg; shift 2 ;;
        --pope-parallel-workers) POPE_PARALLEL_WORKERS="$2"; shift 2 ;;
        --pope-max-samples) POPE_MAX_SAMPLES="$2"; shift 2 ;;
        --pope-use-prepared-query) POPE_USE_PREPARED_QUERY="$2"; shift 2 ;;
        --eval-opd-trace) EVAL_OPD_TRACE="$2"; shift 2 ;;
        --eval-opd-trace-topk) EVAL_OPD_TRACE_TOPK="$2"; shift 2 ;;
        --eval-opd-trace-entropy) EVAL_OPD_TRACE_ENTROPY="$2"; shift 2 ;;
        --eval-opd-trace-score-baseline) EVAL_OPD_TRACE_SCORE_BASELINE="$2"; shift 2 ;;
        --eval-opd-trace-case-analysis) EVAL_OPD_TRACE_CASE_ANALYSIS="$2"; shift 2 ;;
        --eval-opd-trace-max-samples) EVAL_OPD_TRACE_MAX_SAMPLES="$2"; shift 2 ;;
        --vision-benchmark) VISION_BENCHMARK="$2"; shift 2 ;;
        --vision-benchmark-data-dir|--benchmark-data-dir) VISION_BENCHMARK_DATA_DIR="$2"; shift 2 ;;
        --vision-benchmark-auto-download|--benchmark-auto-download) VISION_BENCHMARK_AUTO_DOWNLOAD="$2"; shift 2 ;;
        --vision-benchmark-clean-source|--benchmark-clean-source) VISION_BENCHMARK_CLEAN_SOURCE="$2"; shift 2 ;;
        --vision-benchmark-refresh-prompts|--benchmark-refresh-prompts) VISION_BENCHMARK_REFRESH_PROMPTS="$2"; shift 2 ;;
        --vision-benchmark-output-suffix|--benchmark-output-suffix) VISION_BENCHMARK_OUTPUT_SUFFIX="$2"; shift 2 ;;
        --vision-max-tokens) VISION_MAX_TOKENS="$2"; VISION_MAX_TOKENS_WAS_SET=arg; shift 2 ;;
        --vision-parallel-workers) VISION_PARALLEL_WORKERS="$2"; shift 2 ;;
        --vision-max-retries) VISION_MAX_RETRIES="$2"; shift 2 ;;
        --vision-enable-thinking) VISION_ENABLE_THINKING="$2"; shift 2 ;;
        --vision-rule-only-judge|--rule-only-judge) RULE_ONLY_JUDGE="$2"; shift 2 ;;
        --mcq-extract-mode) MCQ_EXTRACT_MODE="$2"; shift 2 ;;
        --force-vision-eval) FORCE_VISION_EVAL="$2"; shift 2 ;;
        --judge-api-base) JUDGE_API_BASE="$2"; shift 2 ;;
        --judge-api-key) JUDGE_API_KEY="$2"; shift 2 ;;
        --judge-model) JUDGE_MODEL="$2"; shift 2 ;;
        --judge-model-path) JUDGE_MODEL_PATH="$2"; shift 2 ;;
        --judge-max-tokens) JUDGE_MAX_TOKENS="$2"; shift 2 ;;
        --amber-root) AMBER_ROOT="$2"; shift 2 ;;
        --amber-image-root) AMBER_IMAGE_ROOT="$2"; shift 2 ;;
        --amber-eval-type) AMBER_EVAL_TYPE="$2"; shift 2 ;;
        --amber-max-samples) AMBER_MAX_SAMPLES="$2"; shift 2 ;;
        --amber-parallel-workers) AMBER_PARALLEL_WORKERS="$2"; shift 2 ;;
        --amber-official-eval-workers) AMBER_OFFICIAL_EVAL_WORKERS="$2"; shift 2 ;;
        --amber-max-new-tokens-generative) AMBER_MAX_NEW_TOKENS_GENERATIVE="$2"; AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET=arg; shift 2 ;;
        --amber-max-new-tokens-discriminative) AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="$2"; AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET=arg; shift 2 ;;
        --amber-skip-official-eval) AMBER_SKIP_OFFICIAL_EVAL="$2"; shift 2 ;;
        --amber-word-association) AMBER_WORD_ASSOCIATION="$2"; shift 2 ;;
        --amber-safe-words) AMBER_SAFE_WORDS="$2"; shift 2 ;;
        --amber-annotation) AMBER_ANNOTATION="$2"; shift 2 ;;
        --amber-metrics) AMBER_METRICS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

case "${DEGRADATION_MODE:-original}" in
    original)
        ;;
    *)
        echo "Error: --degradation-mode must be original. Got: $DEGRADATION_MODE" >&2
        exit 1
        ;;
esac

is_truthy() {
    [[ "${1:-}" == "True" || "${1:-}" == "true" || "${1:-}" == "1" || "${1:-}" == "yes" || "${1:-}" == "Y" || "${1:-}" == "y" ]]
}

format_prob_tag() {
    "$PYTHON_BIN" - "$1" <<'PY' 2>/dev/null || echo "${1//./p}"
import sys
print(str(int(round(float(sys.argv[1]) * 100))))
PY
}

loss_tag_from_alpha() {
    case "${1:-1.0}" in
        1|1.0|1.00)
            echo "rkl"
            ;;
        0.5|0.50|.5)
            echo "jsd"
            ;;
        0|0.0|0.00)
            echo "fkl"
            ;;
        *)
            echo "a${1:-1.0}"
            ;;
    esac
}

dataset_tag_from_version() {
    case "${1:-full}" in
        full)
            echo "full5k"
            ;;
        quick)
            echo "quick1p5k"
            ;;
        *)
            echo "legacy"
            ;;
    esac
}

default_train_batch_size_for_model() {
    local model_name_lc="$1"
    if [[ "$model_name_lc" == *"8b"* ]]; then
        echo 16
    else
        echo 32
    fi
}

build_experiment_name_from_params() {
    local model_name="${MODEL_NAME:-$(basename "$MODEL_PATH")}"
    local model_name_lc
    model_name_lc="$(echo "$model_name" | tr '[:upper:]' '[:lower:]')"
    local student_ratio="${STUDENT_RATIO:-1.0}"
    local teacher_ratio="${TEACHER_RATIO:-0.75}"
    local teacher_mode="${TEACHER_MODE:-frozen}"
    local alpha="${ALPHA:-1.0}"
    local train_batch_size="${TRAIN_BATCH_SIZE:-$(default_train_batch_size_for_model "$model_name_lc")}"
    local rollout_n="${ROLLOUT_N:-4}"
    local total_epochs="${TOTAL_EPOCHS:-1}"
    local dataset_tag
    local loss_tag
    local name_tags
    local name_suffix

    dataset_tag="$(dataset_tag_from_version "$DATASET_VERSION")"
    loss_tag="$(loss_tag_from_alpha "$alpha")"
    name_tags=("${teacher_mode}" "${loss_tag}")

    if is_truthy "${OPD_SELECTIVE_WEIGHT:-False}"; then
        local sw_mode="${OPD_SELECTIVE_WEIGHT_MODE:-entropy_rkl_bucket}"
        local uncertainty_mode
        if [[ "$sw_mode" == "risk_only_mask" ]]; then
            uncertainty_mode="${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE:-nll}"
            name_tags+=("riskmask" "${uncertainty_mode}" "p$(format_prob_tag "${OPD_RISK_MASK_TOP_P:-0.30}")")
        elif [[ "$sw_mode" == "random_mask" ]]; then
            name_tags+=("randommask" "p$(format_prob_tag "${OPD_RISK_MASK_TOP_P:-0.30}")")
        elif [[ "$sw_mode" == "entropy_mask" ]]; then
            name_tags+=("entropymask" "p$(format_prob_tag "${OPD_RISK_MASK_TOP_P:-0.30}")")
        else
            uncertainty_mode="${OPD_SELECTIVE_WEIGHT_UNCERTAINTY_MODE:-entropy}"
            name_tags+=("sw" "${uncertainty_mode}")
        fi
    fi

    case "${OPD_TOKEN_MASK_PCT:-0.0}" in
        0|0.0|0.00|"")
            ;;
        *)
            name_tags+=("maskp$(format_prob_tag "$OPD_TOKEN_MASK_PCT")")
            ;;
    esac

    name_tags+=("b${train_batch_size}" "rn${rollout_n}" "$dataset_tag" "e${total_epochs}")
    IFS=-
    name_suffix="${name_tags[*]}"
    unset IFS
    echo "Res-OPD-${model_name}-orig-sr${student_ratio}-tr${teacher_ratio}-a${alpha}-${name_suffix}"
}

if [[ ${#OSS_NAMES[@]} -eq 0 && ${#EXPERIMENT_NAMES[@]} -eq 0 ]]; then
    constructed_exp_name="$(build_experiment_name_from_params)"
    EXPERIMENT_NAMES+=("$constructed_exp_name")
    echo "No --experiment-names/--oss-names provided; constructed from params:"
    echo "  ${constructed_exp_name}"
fi

normalize_step_name() {
    local value="$1"
    if [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "global_step_${value}"
    else
        echo "$value"
    fi
}

normalize_eval_mode() {
    local mode
    local normalized=()
    local token
    mode="$(echo "$1" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    IFS=',' read -ra tokens <<< "$mode"
    for token in "${tokens[@]}"; do
        case "$token" in
            all)
                normalized+=(chair pope vision amber)
                ;;
            frequent|coco)
                normalized+=(chair pope)
                ;;
            final|external)
                normalized+=(amber)
                ;;
            vision|vision-opd|aux|auxiliary)
                normalized+=(vision)
                ;;
            mmstar)
                normalized+=(mmstar)
                ;;
            cvbench|cv_bench|cv-bench)
                normalized+=(cv-bench)
                ;;
            chair|pope|amber|mme)
                normalized+=("$token")
                ;;
            "")
                ;;
            *)
                echo "Error: unsupported eval task '$token'. Use chair,pope,coco,mmstar,cv-bench,vision,amber,mme,final,all." >&2
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

resolve_result_version_tag() {
    local exp_name="$1"
    local exp_name_lc
    if [[ -n "$RESULT_VERSION_TAG" ]]; then
        echo "$RESULT_VERSION_TAG"
        return 0
    fi
    exp_name_lc="$(echo "$exp_name" | tr '[:upper:]' '[:lower:]')"
    if [[ "$exp_name_lc" == *"thinking"* || "$VISION_ENABLE_THINKING" == "True" || "$VISION_ENABLE_THINKING" == "true" || "$VISION_ENABLE_THINKING" == "1" ]]; then
        echo "thinking"
    elif [[ "$exp_name_lc" == *"instruct"* ]]; then
        echo "instruct"
    else
        echo "$VERSION_TAG"
    fi
}

get_oss_name_from_exp_name() {
    local exp_name="$1"
    local suffix
    if [[ "$exp_name" == ResOPD_* ]]; then
        echo "$exp_name"
        return 0
    elif [[ "$exp_name" == Res-OPD-Qwen3VL-2B-Instruct-* ]]; then
        # Backward-compatible OSS path for existing Instruct checkpoints.
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

get_exp_name_from_oss_name() {
    local oss_name="$1"
    local suffix
    local exp_suffix
    if [[ "$oss_name" == Res-OPD-* ]]; then
        echo "$oss_name"
        return 0
    elif [[ "$oss_name" != ResOPD_* ]]; then
        echo "$oss_name"
        return 0
    fi

    suffix="${oss_name#ResOPD_}"
    exp_suffix="${suffix//_/-}"

    # Legacy Instruct OSS names strip the model prefix to preserve old paths.
    # New generic names include the model name in the OSS suffix.
    if [[ "$suffix" == orig_* || "$suffix" == s[0-9]* || "$suffix" == sr[0-9.]* || "$suffix" == train* ]]; then
        echo "Res-OPD-Qwen3VL-2B-Instruct-${exp_suffix}"
    else
        echo "Res-OPD-${exp_suffix}"
    fi
}

apply_official_aux_defaults() {
    local benches="$1"
    if [[ ",${benches}," == *",mmstar,"* || ",${benches}," == *",cv-bench,"* ]]; then
        local default_max_tokens="$EVAL_MAX_TOKENS"
        if [[ "${VISION_ENABLE_THINKING}" == "True" || "${VISION_ENABLE_THINKING}" == "true" || "${VISION_ENABLE_THINKING}" == "1" ]]; then
            default_max_tokens="${VISION_THINKING_MAX_TOKENS:-$EVAL_THINKING_MAX_TOKENS}"
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

apply_thinking_eval_defaults() {
    if [[ "${VISION_ENABLE_THINKING}" != "True" && "${VISION_ENABLE_THINKING}" != "true" && "${VISION_ENABLE_THINKING}" != "1" ]]; then
        return 0
    fi
    local thinking_max_tokens="${EVAL_THINKING_MAX_TOKENS:-${EVAL_MAX_TOKENS:-${MAX_RESPONSE_LENGTH:-8192}}}"
    if [[ -z "$CHAIR_MAX_NEW_TOKENS_WAS_SET" ]]; then
        CHAIR_MAX_NEW_TOKENS="$thinking_max_tokens"
    fi
    if [[ -z "$POPE_MAX_NEW_TOKENS_WAS_SET" ]]; then
        POPE_MAX_NEW_TOKENS="$thinking_max_tokens"
    fi
    if [[ -z "$VISION_MAX_TOKENS_WAS_SET" ]]; then
        VISION_MAX_TOKENS="$thinking_max_tokens"
    fi
    if [[ -z "$AMBER_MAX_NEW_TOKENS_GENERATIVE_WAS_SET" ]]; then
        AMBER_MAX_NEW_TOKENS_GENERATIVE="$thinking_max_tokens"
    fi
    if [[ -z "$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE_WAS_SET" ]]; then
        AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="$thinking_max_tokens"
    fi
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
    local target_real
    local base_real
    target_real="$(realpath_for_cleanup "$target")"
    base_real="$(realpath_for_cleanup "$CKPT_BASE")"

    if [[ "$target_real" == "$base_real"/* ]]; then
        rm -rf "$target"
        return 0
    fi

    echo "  ⚠️  Refusing to remove path outside ${base_real}: ${target_real}" >&2
    return 1
}

download_checkpoint() {
    local oss_path="$1"
    local local_ckpt_dir="$2"
    local f

    mkdir -p "$local_ckpt_dir" "${local_ckpt_dir}/actor/huggingface"
    ossutil cp "${oss_path}/model.safetensors" "${local_ckpt_dir}/model.safetensors" -f

    # vLLM reads --model from the checkpoint root, so HF metadata must live
    # beside model.safetensors. Keep actor/huggingface as a compatibility mirror.
    for f in "${HF_FILES[@]}"; do
        if ossutil cp "${oss_path}/${f}" "${local_ckpt_dir}/${f}" -f 2>/dev/null; then
            cp -f "${local_ckpt_dir}/${f}" "${local_ckpt_dir}/actor/huggingface/${f}"
        fi
    done

    if [[ ! -f "${local_ckpt_dir}/config.json" ]]; then
        echo "Error: config.json was not downloaded to ${local_ckpt_dir}." >&2
        echo "Check OSS path: ${oss_path}/config.json" >&2
        return 1
    fi
}

verify_oss_backup() {
    local oss_path="$1"
    if ossutil stat "${oss_path}/model.safetensors" > /dev/null 2>&1; then
        return 0
    else
        echo "  ⚠️  OSS backup not found at ${oss_path}/model.safetensors" >&2
        return 1
    fi
}

cleanup_checkpoint() {
    local local_ckpt_dir="$1"
    local oss_path="$2"

    # Only delete local checkpoint if OSS backup is confirmed
    if [[ -n "$oss_path" ]] && verify_oss_backup "$oss_path"; then
        echo "  OSS backup verified, removing local checkpoint directory ..."
        safe_remove_checkpoint_dir "$local_ckpt_dir"
        # Also remove parent experiment dir if empty
        local parent_dir
        parent_dir="$(dirname "$local_ckpt_dir")"
        if [[ -d "$parent_dir" ]] && [ -z "$(ls -A "$parent_dir" 2>/dev/null)" ]; then
            rmdir "$parent_dir" 2>/dev/null || true
        fi
    else
        echo "  ⚠️  Skipping local cleanup: OSS backup not verified. Keeping local checkpoint." >&2
    fi
}

list_oss_steps() {
    local oss_exp_path="${1%/}"
    local output
    local steps

    if ! output="$(ossutil ls "${oss_exp_path}/" 2>&1)"; then
        echo "Error: failed to list OSS checkpoints under ${oss_exp_path}/" >&2
        echo "$output" >&2
        return 1
    fi

    steps="$(
        printf '%s\n' "$output" \
            | grep -oE 'global_step_[0-9]+' \
            | sort -t_ -k3,3n -u || true
    )"
    if [[ -z "$steps" ]]; then
        echo "Error: no global_step_* checkpoints found under ${oss_exp_path}/" >&2
        echo "Pass --step explicitly if the OSS layout is nonstandard." >&2
        return 1
    fi
    printf '%s\n' "$steps"
}

resolve_steps_for_experiment() {
    local oss_exp_path="$1"
    if [[ -n "$STEP" ]]; then
        normalize_step_name "$STEP"
        return 0
    fi
    list_oss_steps "$oss_exp_path"
}

infer_eval_spec() {
    local exp_name="$1"
    local inferred_mode="original"
    local inferred_student_px="0"
    local inferred_teacher_px="$TARGET_PX"
    local inferred_student_ratio="1.0"
    local inferred_teacher_ratio="1.0"

    if [[ "$exp_name" =~ -orig-sr([0-9.]+)-tr([0-9.]+) ]]; then
        inferred_student_px="0"
        inferred_student_ratio="${BASH_REMATCH[1]}"
        inferred_teacher_ratio="${BASH_REMATCH[2]}"
    elif [[ "$exp_name" =~ -s([0-9]+)-t([0-9]+) ]]; then
        inferred_student_px="${BASH_REMATCH[1]}"
        inferred_teacher_px="${BASH_REMATCH[2]}"
        inferred_student_ratio="1.0"
    elif [[ "$exp_name" =~ -s([0-9]+)(-|_) ]]; then
        inferred_student_px="${BASH_REMATCH[1]}"
        inferred_student_ratio="1.0"
    fi

    echo "${DEGRADATION_MODE:-$inferred_mode}|${STUDENT_PX:-$inferred_student_px}|${TEACHER_PX:-$inferred_teacher_px}|${TARGET_PX}|${STUDENT_RATIO:-$inferred_student_ratio}|${TEACHER_RATIO:-$inferred_teacher_ratio}"
}

eval_results_exist() {
    local result_dir="$1"
    local ok=0
    if has_eval_task "$EVAL_MODE" "chair"; then
        if ! ls "${result_dir}/"*/chair_metrics.json 2>/dev/null | grep -q .; then
            ok=1
        fi
        if [[ "$EVAL_OPD_TRACE" == "True" || "$EVAL_OPD_TRACE" == "true" || "$EVAL_OPD_TRACE" == "1" ]]; then
            if ! ls "${result_dir}/"*/opd_eval_trace_summary.json 2>/dev/null | grep -q .; then
                ok=1
            fi
        fi
    fi
    if has_eval_task "$EVAL_MODE" "pope"; then
        if ! ls "${result_dir}/"*/pope/pope_summary.json 2>/dev/null | grep -q .; then
            ok=1
        fi
    fi
    if has_eval_task "$EVAL_MODE" "amber"; then
        if ! ls "${result_dir}/"*/amber/amber_metrics.json 2>/dev/null | grep -q .; then
            ok=1
        fi
    fi
    if has_eval_task "$EVAL_MODE" "mme"; then
        if ! [[ -f "${result_dir}/final_hallucination/mme_metrics.json" ]]; then
            ok=1
        fi
    fi
    if has_vision_eval_task "$EVAL_MODE"; then
        if [[ "$FORCE_VISION_EVAL" == "True" || "$FORCE_VISION_EVAL" == "true" || "$FORCE_VISION_EVAL" == "1" ]]; then
            ok=1
        else
            if ! "$PYTHON_BIN" "$VISION_STATUS_SCRIPT" \
                --result-root "$result_dir" \
                --benchmarks "$EFFECTIVE_VISION_BENCHMARK" \
                --benchmark-data-dir "$VISION_BENCHMARK_DATA_DIR" \
                --output-suffix "$VISION_BENCHMARK_OUTPUT_SUFFIX" >/dev/null 2>&1; then
                ok=1
            fi
        fi
    fi
    return $ok
}

EVAL_MODE="$(normalize_eval_mode "$EVAL_MODE")"
EFFECTIVE_VISION_BENCHMARK="$(resolve_vision_benchmarks "$EVAL_MODE")"
if [[ -z "$VISION_ENABLE_THINKING" ]]; then
    for candidate_name in "${EXPERIMENT_NAMES[@]:-}" "${OSS_NAMES[@]:-}" "${LOCAL_NAMES[@]:-}"; do
        candidate_lc="$(echo "$candidate_name" | tr '[:upper:]' '[:lower:]')"
        if [[ "$candidate_lc" == *"thinking"* ]]; then
            VISION_ENABLE_THINKING="True"
            break
        fi
    done
fi
apply_official_aux_defaults "$EFFECTIVE_VISION_BENCHMARK"
apply_thinking_eval_defaults
if is_truthy "$EVAL_SHARDED"; then
    EVAL_BACKEND="sharded"
fi
case "$EVAL_BACKEND" in
    single|sharded)
        ;;
    *)
        echo "Error: --eval-backend must be single or sharded. Got: ${EVAL_BACKEND}" >&2
        exit 1
        ;;
esac
if [[ "$EVAL_BACKEND" == "sharded" ]]; then
    if ! [[ "$EVAL_SHARD_COUNT" =~ ^[0-9]+$ ]] || [[ "$EVAL_SHARD_COUNT" -le 0 ]]; then
        echo "Error: EVAL_SHARD_COUNT must be a positive integer. Got: ${EVAL_SHARD_COUNT}" >&2
        exit 1
    fi
fi
if [[ -z "$EVAL_MODE" ]]; then
    echo "Error: --eval-mode expanded to empty. Use chair,pope,coco,mmstar,cv-bench,vision,amber,mme,final,all." >&2
    exit 1
fi
if [[ ${#EXPERIMENT_NAMES[@]} -gt 0 ]]; then
    NUM_EXPERIMENTS=${#EXPERIMENT_NAMES[@]}
else
    NUM_EXPERIMENTS=${#OSS_NAMES[@]}
fi

echo "=========================================="
echo " Batch Eval from OSS"
if [[ -n "$STEP" ]]; then
    echo " Step: $(normalize_step_name "$STEP")"
else
    echo " Step: all OSS global_step_* checkpoints, serial per experiment"
fi
echo " Student spec: auto from experiment name unless overridden"
echo " Target PX: ${TARGET_PX}"
echo " Version tag: ${VERSION_TAG}"
echo " Result tag override: ${RESULT_VERSION_TAG:-<auto by model profile>}"
echo " Eval mode: ${EVAL_MODE}"
echo " Eval backend: ${EVAL_BACKEND}"
echo " Enable thinking: ${VISION_ENABLE_THINKING:-<auto/off>}"
echo " Max tokens: chair=${CHAIR_MAX_NEW_TOKENS}, pope=${POPE_MAX_NEW_TOKENS}, vision=${VISION_MAX_TOKENS}, amber_gen=${AMBER_MAX_NEW_TOKENS_GENERATIVE}, amber_disc=${AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE}"
if [[ "$EVAL_BACKEND" == "sharded" ]]; then
    echo " Sharded eval: shards=${EVAL_SHARD_COUNT}, gpu_list=${GPU_LIST}, keep_shards=${SHARDED_EVAL_KEEP_SHARDS}"
fi
if has_vision_eval_task "$EVAL_MODE"; then
    echo " Vision benchmarks: ${EFFECTIVE_VISION_BENCHMARK}"
    echo " Vision data dir: ${VISION_BENCHMARK_DATA_DIR}"
    echo " Vision auto download: ${VISION_BENCHMARK_AUTO_DOWNLOAD}"
    echo " Vision clean source: ${VISION_BENCHMARK_CLEAN_SOURCE}"
    echo " Vision refresh prompts: ${VISION_BENCHMARK_REFRESH_PROMPTS}"
    echo " Vision output suffix: ${VISION_BENCHMARK_OUTPUT_SUFFIX}"
    echo " vLLM port cleanup: ${VLLM_PORT_CLEANUP}"
    echo " Rule-only judge: ${RULE_ONLY_JUDGE}"
    echo " MCQ extract mode: ${MCQ_EXTRACT_MODE}"
    echo " Force vision eval: ${FORCE_VISION_EVAL}"
fi
echo " CHAIR logprobs: ${CHAIR_SAVE_LOGPROBS} (top=${CHAIR_TOP_LOGPROBS})"
echo " OPD eval trace: ${EVAL_OPD_TRACE} (topk=${EVAL_OPD_TRACE_TOPK}, entropy=${EVAL_OPD_TRACE_ENTROPY})"
echo " AMBER official eval workers: ${AMBER_OFFICIAL_EVAL_WORKERS}"
echo " Experiments: ${NUM_EXPERIMENTS}"
echo " Started at: $(date)"
echo "=========================================="

VLLM_BASE_PORT="${VLLM_BASE_PORT:-8000}"

for ((idx = 0; idx < NUM_EXPERIMENTS; idx++)); do
    if [[ $idx -lt ${#EXPERIMENT_NAMES[@]} && -n "${EXPERIMENT_NAMES[$idx]}" ]]; then
        local_exp_name="${EXPERIMENT_NAMES[$idx]}"
        if [[ $idx -lt ${#OSS_NAMES[@]} && -n "${OSS_NAMES[$idx]}" ]]; then
            oss_name="${OSS_NAMES[$idx]}"
        else
            oss_name="$(get_oss_name_from_exp_name "$local_exp_name")"
        fi
    else
        oss_name="${OSS_NAMES[$idx]}"
        # Derive local experiment name with fallback:
        #   1. Explicit --local-names (if provided for this index)
        #   2. OSS-name reverse mapping used by the checkpoint uploader
        if [[ $idx -lt ${#LOCAL_NAMES[@]} && -n "${LOCAL_NAMES[$idx]}" ]]; then
            local_exp_name="${LOCAL_NAMES[$idx]}"
        else
            local_exp_name="$(get_exp_name_from_oss_name "$oss_name")"
        fi
    fi

    # Increment port for each experiment to avoid vLLM port conflicts
    export VLLM_PORT=$((VLLM_BASE_PORT + idx))

    # Set per-experiment student ratio if provided
    if [[ $idx -lt ${#STUDENT_RATIOS[@]} && -n "${STUDENT_RATIOS[$idx]}" ]]; then
        export STUDENT_RATIO="${STUDENT_RATIOS[$idx]}"
        echo "  Student ratio: ${STUDENT_RATIO}"
    fi
    result_version_tag="$(resolve_result_version_tag "$local_exp_name")"
    IFS='|' read -r effective_degradation_mode effective_student_px effective_teacher_px effective_target_px effective_student_ratio effective_teacher_ratio < <(
        infer_eval_spec "$local_exp_name"
    )

    oss_exp_path="${OSS_BASE}/${oss_name}"
    mapfile -t steps_to_eval < <(resolve_steps_for_experiment "$oss_exp_path")
    if [[ ${#steps_to_eval[@]} -eq 0 ]]; then
        echo "Error: no checkpoint steps resolved for ${oss_name}" >&2
        exit 1
    fi
    echo ""
    echo "=========================================="
    echo " Experiment: ${oss_name}"
    echo " Local exp:  ${local_exp_name}"
    echo " Result tag: ${result_version_tag}"
    echo " Steps:      ${steps_to_eval[*]}"
    echo " Student: mode=${effective_degradation_mode} px=${effective_student_px} target=${effective_target_px} ratio=${effective_student_ratio}"
    echo " Teacher: px=${effective_teacher_px} ratio=${effective_teacher_ratio}"
    echo "=========================================="

    for step in "${steps_to_eval[@]}"; do
        oss_path="${oss_exp_path}/${step}"
        local_ckpt_dir="${CKPT_BASE}/${local_exp_name}/${step}"

        echo ""
        echo "=========================================="
        echo " Processing: ${oss_name}/${step}"
        echo " Local exp: ${local_exp_name}_${step}"
        echo " OSS: ${oss_path}"
        echo "=========================================="

        # Check if eval results already exist
        if [[ "$DATASET_VERSION" == "full" ]]; then
            existing_result_root="${RES_OPD_ROOT}/eval_results/${result_version_tag}/full/${local_exp_name}_${step}"
        else
            existing_result_root="${RES_OPD_ROOT}/eval_results/${result_version_tag}/${local_exp_name}_${step}"
        fi
        if eval_results_exist "$existing_result_root"; then
            echo "⚠️  Eval results already exist for ${step}, skipping."
            continue
        fi

        # Step 1: Download checkpoint from OSS
        echo "[1/4] Downloading from OSS ..."
        download_checkpoint "$oss_path" "$local_ckpt_dir"
        echo "  ✅ Download complete"

        # Step 2: Run evaluation
        echo "[2/4] Running evaluation ..."
    if [[ "$EVAL_BACKEND" == "sharded" ]]; then
        if has_eval_task "$EVAL_MODE" "mme"; then
            echo "Error: sharded backend does not support legacy MME script. Run MME separately or use --eval-backend single." >&2
            exit 1
        fi
        echo "  Running sharded eval (${EVAL_MODE}) ..."
        sharded_env=(
            DATASET_VERSION="$DATASET_VERSION"
            CLEANUP_LOCAL_CKPT=False
            VLLM_PORT_CLEANUP="$VLLM_PORT_CLEANUP"
            VLLM_PORT_CLEANUP_WAIT="$VLLM_PORT_CLEANUP_WAIT"
            EVAL_MAX_TOKENS="$EVAL_MAX_TOKENS"
            EVAL_THINKING_MAX_TOKENS="$EVAL_THINKING_MAX_TOKENS"
            VLLM_BASE_PORT="$((VLLM_BASE_PORT + idx * EVAL_SHARD_COUNT))"
            EVAL_SHARD_COUNT="$EVAL_SHARD_COUNT"
            GPU_LIST="$GPU_LIST"
            SHARDED_EVAL_KEEP_SHARDS="$SHARDED_EVAL_KEEP_SHARDS"
            DEGRADATION_MODE="$effective_degradation_mode"
            STUDENT_RATIO="$effective_student_ratio"
            TEACHER_RATIO="$effective_teacher_ratio"
            TEACHER_PX="$effective_teacher_px"
            TARGET_PX="$effective_target_px"
            CHAIR_MAX_NEW_TOKENS="$CHAIR_MAX_NEW_TOKENS"
            CHAIR_PARALLEL_WORKERS="$CHAIR_PARALLEL_WORKERS"
            CHAIR_MAX_SAMPLES="$CHAIR_MAX_SAMPLES"
            CHAIR_SAVE_LOGPROBS="$CHAIR_SAVE_LOGPROBS"
            CHAIR_TOP_LOGPROBS="$CHAIR_TOP_LOGPROBS"
            POPE_MAX_NEW_TOKENS="$POPE_MAX_NEW_TOKENS"
            POPE_PARALLEL_WORKERS="$POPE_PARALLEL_WORKERS"
            POPE_MAX_SAMPLES="$POPE_MAX_SAMPLES"
            POPE_USE_PREPARED_QUERY="$POPE_USE_PREPARED_QUERY"
            EVAL_OPD_TRACE="$EVAL_OPD_TRACE"
            EVAL_OPD_TRACE_TOPK="$EVAL_OPD_TRACE_TOPK"
            EVAL_OPD_TRACE_ENTROPY="$EVAL_OPD_TRACE_ENTROPY"
            EVAL_OPD_TRACE_SCORE_BASELINE="$EVAL_OPD_TRACE_SCORE_BASELINE"
            EVAL_OPD_TRACE_CASE_ANALYSIS="$EVAL_OPD_TRACE_CASE_ANALYSIS"
            EVAL_OPD_TRACE_MAX_SAMPLES="$EVAL_OPD_TRACE_MAX_SAMPLES"
            VISION_BENCHMARK="$EFFECTIVE_VISION_BENCHMARK"
            VISION_BENCHMARK_DATA_DIR="$VISION_BENCHMARK_DATA_DIR"
            VISION_BENCHMARK_AUTO_DOWNLOAD="$VISION_BENCHMARK_AUTO_DOWNLOAD"
            VISION_BENCHMARK_CLEAN_SOURCE="$VISION_BENCHMARK_CLEAN_SOURCE"
            VISION_BENCHMARK_REFRESH_PROMPTS="$VISION_BENCHMARK_REFRESH_PROMPTS"
            VISION_BENCHMARK_OUTPUT_SUFFIX="$VISION_BENCHMARK_OUTPUT_SUFFIX"
            VISION_MAX_TOKENS="$VISION_MAX_TOKENS"
            VISION_PARALLEL_WORKERS="$VISION_PARALLEL_WORKERS"
            VISION_MAX_RETRIES="$VISION_MAX_RETRIES"
            RULE_ONLY_JUDGE="$RULE_ONLY_JUDGE"
            MCQ_EXTRACT_MODE="$MCQ_EXTRACT_MODE"
            JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS"
            AMBER_ROOT="$AMBER_ROOT"
            AMBER_IMAGE_ROOT="$AMBER_IMAGE_ROOT"
            AMBER_EVAL_TYPE="$AMBER_EVAL_TYPE"
            AMBER_MAX_SAMPLES="$AMBER_MAX_SAMPLES"
            AMBER_PARALLEL_WORKERS="$AMBER_PARALLEL_WORKERS"
            AMBER_OFFICIAL_EVAL_WORKERS="$AMBER_OFFICIAL_EVAL_WORKERS"
            AMBER_MAX_NEW_TOKENS_GENERATIVE="$AMBER_MAX_NEW_TOKENS_GENERATIVE"
            AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE"
            AMBER_SKIP_OFFICIAL_EVAL="$AMBER_SKIP_OFFICIAL_EVAL"
            AMBER_WORD_ASSOCIATION="$AMBER_WORD_ASSOCIATION"
            AMBER_SAFE_WORDS="$AMBER_SAFE_WORDS"
            AMBER_ANNOTATION="$AMBER_ANNOTATION"
            AMBER_METRICS="$AMBER_METRICS"
            RESULT_VERSION_TAG="$result_version_tag"
        )
        [[ -n "$VISION_ENABLE_THINKING" ]] && sharded_env+=(VISION_ENABLE_THINKING="$VISION_ENABLE_THINKING")
        [[ -n "${JUDGE_API_BASE:-}" ]] && sharded_env+=(JUDGE_API_BASE="$JUDGE_API_BASE")
        [[ -n "${JUDGE_API_KEY:-}" ]] && sharded_env+=(JUDGE_API_KEY="$JUDGE_API_KEY")
        [[ -n "${JUDGE_MODEL:-}" ]] && sharded_env+=(JUDGE_MODEL="$JUDGE_MODEL")
        [[ -n "${JUDGE_MODEL_PATH:-}" ]] && sharded_env+=(JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH")
        env "${sharded_env[@]}" bash "$SHARDED_EVAL_SCRIPT" "$local_ckpt_dir" "$effective_student_px" "$VERSION_TAG" "$EVAL_MODE"
    else
    coco_tasks=()
    has_eval_task "$EVAL_MODE" "chair" && coco_tasks+=(chair)
    has_eval_task "$EVAL_MODE" "pope" && coco_tasks+=(pope)
    if [[ ${#coco_tasks[@]} -gt 0 ]]; then
        coco_mode="$(IFS=','; echo "${coco_tasks[*]}")"
        echo "  Running local COCO eval (${coco_mode}) ..."
        DATASET_VERSION="$DATASET_VERSION" \
        CLEANUP_LOCAL_CKPT=False \
        VLLM_PORT_CLEANUP="$VLLM_PORT_CLEANUP" \
        VLLM_PORT_CLEANUP_WAIT="$VLLM_PORT_CLEANUP_WAIT" \
        EVAL_MAX_TOKENS="$EVAL_MAX_TOKENS" \
        EVAL_THINKING_MAX_TOKENS="$EVAL_THINKING_MAX_TOKENS" \
        DEGRADATION_MODE="$effective_degradation_mode" \
        STUDENT_RATIO="$effective_student_ratio" \
        TEACHER_RATIO="$effective_teacher_ratio" \
        TEACHER_PX="$effective_teacher_px" \
        TARGET_PX="$effective_target_px" \
        CHAIR_MAX_NEW_TOKENS="$CHAIR_MAX_NEW_TOKENS" \
        CHAIR_PARALLEL_WORKERS="$CHAIR_PARALLEL_WORKERS" \
        CHAIR_MAX_SAMPLES="$CHAIR_MAX_SAMPLES" \
        CHAIR_SAVE_LOGPROBS="$CHAIR_SAVE_LOGPROBS" \
        CHAIR_TOP_LOGPROBS="$CHAIR_TOP_LOGPROBS" \
        POPE_MAX_NEW_TOKENS="$POPE_MAX_NEW_TOKENS" \
        POPE_PARALLEL_WORKERS="$POPE_PARALLEL_WORKERS" \
        POPE_MAX_SAMPLES="$POPE_MAX_SAMPLES" \
        POPE_USE_PREPARED_QUERY="$POPE_USE_PREPARED_QUERY" \
        VISION_ENABLE_THINKING="$VISION_ENABLE_THINKING" \
        EVAL_OPD_TRACE="$EVAL_OPD_TRACE" \
        EVAL_OPD_TRACE_TOPK="$EVAL_OPD_TRACE_TOPK" \
        EVAL_OPD_TRACE_ENTROPY="$EVAL_OPD_TRACE_ENTROPY" \
        EVAL_OPD_TRACE_SCORE_BASELINE="$EVAL_OPD_TRACE_SCORE_BASELINE" \
        EVAL_OPD_TRACE_CASE_ANALYSIS="$EVAL_OPD_TRACE_CASE_ANALYSIS" \
        EVAL_OPD_TRACE_MAX_SAMPLES="$EVAL_OPD_TRACE_MAX_SAMPLES" \
        RESULT_VERSION_TAG="$result_version_tag" \
            bash "$EVAL_SCRIPT" "$local_ckpt_dir" "$effective_student_px" "$VERSION_TAG" "$coco_mode"
    fi
    if has_vision_eval_task "$EVAL_MODE"; then
        echo "  Running Vision-OPD auxiliary eval (${EFFECTIVE_VISION_BENCHMARK}) ..."
        vision_env=(
            DATASET_VERSION="$DATASET_VERSION"
            CLEANUP_LOCAL_CKPT=False
            VLLM_PORT_CLEANUP="$VLLM_PORT_CLEANUP"
            VLLM_PORT_CLEANUP_WAIT="$VLLM_PORT_CLEANUP_WAIT"
            EVAL_MAX_TOKENS="$EVAL_MAX_TOKENS"
            EVAL_THINKING_MAX_TOKENS="$EVAL_THINKING_MAX_TOKENS"
            DEGRADATION_MODE="$effective_degradation_mode"
            STUDENT_RATIO="$effective_student_ratio"
            TEACHER_RATIO="$effective_teacher_ratio"
            TEACHER_PX="$effective_teacher_px"
            TARGET_PX="$effective_target_px"
            VISION_BENCHMARK="$EFFECTIVE_VISION_BENCHMARK"
            VISION_BENCHMARK_DATA_DIR="$VISION_BENCHMARK_DATA_DIR"
            VISION_BENCHMARK_AUTO_DOWNLOAD="$VISION_BENCHMARK_AUTO_DOWNLOAD"
            VISION_BENCHMARK_CLEAN_SOURCE="$VISION_BENCHMARK_CLEAN_SOURCE"
            VISION_BENCHMARK_REFRESH_PROMPTS="$VISION_BENCHMARK_REFRESH_PROMPTS"
            VISION_BENCHMARK_OUTPUT_SUFFIX="$VISION_BENCHMARK_OUTPUT_SUFFIX"
            VISION_MAX_TOKENS="$VISION_MAX_TOKENS"
            VISION_PARALLEL_WORKERS="$VISION_PARALLEL_WORKERS"
            VISION_MAX_RETRIES="$VISION_MAX_RETRIES"
            RULE_ONLY_JUDGE="$RULE_ONLY_JUDGE"
            MCQ_EXTRACT_MODE="$MCQ_EXTRACT_MODE"
            JUDGE_MAX_TOKENS="$JUDGE_MAX_TOKENS"
            RESULT_VERSION_TAG="$result_version_tag"
        )
        [[ -n "$VISION_ENABLE_THINKING" ]] && vision_env+=(VISION_ENABLE_THINKING="$VISION_ENABLE_THINKING")
        [[ -n "${JUDGE_API_BASE:-}" ]] && vision_env+=(JUDGE_API_BASE="$JUDGE_API_BASE")
        [[ -n "${JUDGE_API_KEY:-}" ]] && vision_env+=(JUDGE_API_KEY="$JUDGE_API_KEY")
        [[ -n "${JUDGE_MODEL:-}" ]] && vision_env+=(JUDGE_MODEL="$JUDGE_MODEL")
        [[ -n "${JUDGE_MODEL_PATH:-}" ]] && vision_env+=(JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH")
        env "${vision_env[@]}" bash "$EVAL_SCRIPT" "$local_ckpt_dir" "$effective_student_px" "$VERSION_TAG" "vision"
    fi
    if has_eval_task "$EVAL_MODE" "amber"; then
        echo "  Running AMBER ..."
        eval_env=(
            DATASET_VERSION="$DATASET_VERSION"
            CLEANUP_LOCAL_CKPT=False
            EVAL_MAX_TOKENS="$EVAL_MAX_TOKENS"
            EVAL_THINKING_MAX_TOKENS="$EVAL_THINKING_MAX_TOKENS"
            DEGRADATION_MODE="$effective_degradation_mode"
            STUDENT_RATIO="$effective_student_ratio"
            TARGET_PX="$effective_target_px"
            AMBER_ROOT="$AMBER_ROOT"
            AMBER_IMAGE_ROOT="$AMBER_IMAGE_ROOT"
            AMBER_EVAL_TYPE="$AMBER_EVAL_TYPE"
            AMBER_MAX_SAMPLES="$AMBER_MAX_SAMPLES"
            AMBER_PARALLEL_WORKERS="$AMBER_PARALLEL_WORKERS"
            AMBER_OFFICIAL_EVAL_WORKERS="$AMBER_OFFICIAL_EVAL_WORKERS"
            AMBER_MAX_NEW_TOKENS_GENERATIVE="$AMBER_MAX_NEW_TOKENS_GENERATIVE"
            AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE="$AMBER_MAX_NEW_TOKENS_DISCRIMINATIVE"
            AMBER_SKIP_OFFICIAL_EVAL="$AMBER_SKIP_OFFICIAL_EVAL"
            AMBER_WORD_ASSOCIATION="$AMBER_WORD_ASSOCIATION"
            AMBER_SAFE_WORDS="$AMBER_SAFE_WORDS"
            AMBER_ANNOTATION="$AMBER_ANNOTATION"
            AMBER_METRICS="$AMBER_METRICS"
            RESULT_VERSION_TAG="$result_version_tag"
        )
        [[ -n "$VISION_ENABLE_THINKING" ]] && eval_env+=(VISION_ENABLE_THINKING="$VISION_ENABLE_THINKING")
        env "${eval_env[@]}" bash "$EVAL_SCRIPT" "$local_ckpt_dir" "$effective_student_px" "$VERSION_TAG" "amber"
    fi
    if has_eval_task "$EVAL_MODE" "mme"; then
        echo "  Running classic MME perception ..."
        STUDENT_PX="$effective_student_px" \
        TARGET_PX="$effective_target_px" \
        DEGRADATION_MODE="$effective_degradation_mode" \
        STUDENT_RATIO="$effective_student_ratio" \
            bash "$MME_SCRIPT" "$local_ckpt_dir" "$VERSION_TAG"
    fi
    fi
    echo "  ✅ Evaluation complete"

    # Step 3: Delete local checkpoint after verifying OSS backup
    echo "[3/4] Cleaning up local checkpoint (after OSS verification) ..."
    cleanup_checkpoint "$local_ckpt_dir" "$oss_path"
    echo "  ✅ Local checkpoint cleaned up"

    # Step 4: Verify eval results
    echo "[4/4] Verifying eval results ..."
    if [[ "$DATASET_VERSION" == "full" ]]; then
        result_dir="${RES_OPD_ROOT}/eval_results/${result_version_tag}/full/${local_exp_name}_${step}"
    else
        result_dir="${RES_OPD_ROOT}/eval_results/${result_version_tag}/${local_exp_name}_${step}"
    fi
    if eval_results_exist "$result_dir"; then
        echo "  ✅ Eval results saved:"
        find "${result_dir}" \( -name "chair_metrics.json" -o -name "pope_summary.json" -o -name "amber_metrics.json" -o -name "mme_metrics.json" -o -name "*_answer.jsonl" \) -exec echo "    {}" \;
    else
        echo "  ⚠️  No eval results found!"
        if has_eval_task "$EVAL_MODE" "vision"; then
            "$PYTHON_BIN" "$VISION_STATUS_SCRIPT" \
                --result-root "$result_dir" \
                --benchmarks "$EFFECTIVE_VISION_BENCHMARK" \
                --benchmark-data-dir "$VISION_BENCHMARK_DATA_DIR" \
                --output-suffix "$VISION_BENCHMARK_OUTPUT_SUFFIX" || true
        fi
    fi

    echo ""
    echo "✅ Done: ${oss_name}/${step}"
    done

    echo ""
    echo "✅ Done experiment: ${oss_name}"
done

echo ""
echo "=========================================="
echo " All evaluations completed!"
echo " Finished at: $(date)"
echo "=========================================="

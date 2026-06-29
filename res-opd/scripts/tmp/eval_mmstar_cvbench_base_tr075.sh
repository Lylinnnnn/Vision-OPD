#!/usr/bin/env bash
set -euo pipefail

# Temporary helper: evaluate base + tr0.75 variants on MMStar/CV-Bench.
# Results are written under each checkpoint's normal eval_results directory:
#   <ckpt_result_dir>/mmstar/{model_answer,judge}/...
#   <ckpt_result_dir>/cv-bench/{model_answer,judge}/...

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}"
RUN_BASE="${RUN_BASE:-True}"

STEP="${STEP:-global_step_39}"
VERSION_TAG="${VERSION_TAG:-latest}"
VLLM_BASE_PORT="${VLLM_BASE_PORT:-8020}"
VISION_BENCHMARK="${VISION_BENCHMARK:-mmstar,cv-bench}"
VISION_BENCHMARK_DATA_DIR="${VISION_BENCHMARK_DATA_DIR:-/home/liuyanlin.lyl/notebook/data}"
VISION_BENCHMARK_AUTO_DOWNLOAD="${VISION_BENCHMARK_AUTO_DOWNLOAD:-True}"
VISION_BENCHMARK_CLEAN_SOURCE="${VISION_BENCHMARK_CLEAN_SOURCE:-True}"
RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-True}"
VISION_PARALLEL_WORKERS="${VISION_PARALLEL_WORKERS:-128}"
VISION_MAX_RETRIES="${VISION_MAX_RETRIES:-3}"
VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-32768}"

TR075_RKL_OSS="${TR075_RKL_OSS:-ResOPD_orig_sr1.0_tr0.75_a1.0_frozen_rkl_full5k-e1}"
TR075_SW_OSS="${TR075_SW_OSS:-ResOPD_orig_sr1.0_tr0.75_a1.0_frozen_rkl_sw_full5k-e1}"
TR075_SW075_OSS="${TR075_SW075_OSS:-ResOPD_orig_sr1.0_tr0.75_a1.0_frozen_rkl_sw075_full5k-e1}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/home/liuyanlin.lyl/notebook/data/hf_cache}"

benchmark_json_name() {
    case "$1" in
        mmstar) echo "mmstar.json" ;;
        cv-bench) echo "cv_bench.json" ;;
        *)
            echo "Error: this temporary script only preflights mmstar/cv-bench. Got: $1" >&2
            return 1
            ;;
    esac
}

is_truthy() {
    [[ "$1" == "True" || "$1" == "true" || "$1" == "1" || "$1" == "yes" || "$1" == "Y" || "$1" == "y" ]]
}

preflight_benchmark_data() {
    echo ""
    echo "[data] Preparing benchmark data before starting any model server ..."
    mkdir -p "$VISION_BENCHMARK_DATA_DIR"

    local old_ifs="$IFS"
    local benchmarks=()
    local bench
    IFS=',' read -r -a benchmarks <<< "$VISION_BENCHMARK"
    IFS="$old_ifs"

    for bench in "${benchmarks[@]}"; do
        bench="$(echo "$bench" | xargs)"
        [[ -z "$bench" ]] && continue

        local json_name
        json_name="$(benchmark_json_name "$bench")"
        local json_path="${VISION_BENCHMARK_DATA_DIR}/${json_name}"
        local prepare_args=(
            --benchmark "$bench"
            --data_dir "$VISION_BENCHMARK_DATA_DIR"
        )
        if is_truthy "$VISION_BENCHMARK_CLEAN_SOURCE"; then
            prepare_args+=(--clean-source)
        fi

        if [[ -s "$json_path" ]]; then
            echo "  Found ${bench}: ${json_path}"
            if is_truthy "$VISION_BENCHMARK_CLEAN_SOURCE"; then
                /home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 eval/prepare_data.py "${prepare_args[@]}"
            fi
            continue
        fi

        if ! is_truthy "$VISION_BENCHMARK_AUTO_DOWNLOAD"; then
            echo "Error: ${json_path} is missing and VISION_BENCHMARK_AUTO_DOWNLOAD=${VISION_BENCHMARK_AUTO_DOWNLOAD}" >&2
            exit 1
        fi

        echo "  Preparing ${bench}: ${json_path}"
        /home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 eval/prepare_data.py "${prepare_args[@]}"

        if [[ ! -s "$json_path" ]]; then
            echo "Error: failed to prepare ${json_path}" >&2
            exit 1
        fi
    done
}

echo "============================================================"
echo " MMStar/CV-Bench batch eval"
echo "============================================================"
echo "Benchmarks:     ${VISION_BENCHMARK}"
echo "Data dir:       ${VISION_BENCHMARK_DATA_DIR}"
echo "Auto download:  ${VISION_BENCHMARK_AUTO_DOWNLOAD}"
echo "Clean source:   ${VISION_BENCHMARK_CLEAN_SOURCE}"
echo "Version tag:    ${VERSION_TAG}"
echo "Step:           ${STEP}"
echo "Rule-only:      ${RULE_ONLY_JUDGE}"
echo "HF endpoint:    ${HF_ENDPOINT}"
echo "HF home:        ${HF_HOME}"
echo "============================================================"

preflight_benchmark_data

if [[ "$RUN_BASE" == "True" || "$RUN_BASE" == "true" || "$RUN_BASE" == "1" ]]; then
    if [[ ! -d "$BASE_MODEL_PATH" ]]; then
        echo "Error: BASE_MODEL_PATH does not exist: ${BASE_MODEL_PATH}" >&2
        echo "Set BASE_MODEL_PATH=... or RUN_BASE=False." >&2
        exit 1
    fi
    echo ""
    echo "[base] Evaluating ${BASE_MODEL_PATH}"
    VLLM_PORT="$VLLM_BASE_PORT" \
    CLEANUP_LOCAL_CKPT=False \
    DATASET_VERSION=full \
    DEGRADATION_MODE=original \
    STUDENT_RATIO=1.0 \
    VISION_BENCHMARK="$VISION_BENCHMARK" \
    VISION_BENCHMARK_DATA_DIR="$VISION_BENCHMARK_DATA_DIR" \
    VISION_BENCHMARK_AUTO_DOWNLOAD="$VISION_BENCHMARK_AUTO_DOWNLOAD" \
    VISION_BENCHMARK_CLEAN_SOURCE="$VISION_BENCHMARK_CLEAN_SOURCE" \
    VISION_PARALLEL_WORKERS="$VISION_PARALLEL_WORKERS" \
    VISION_MAX_RETRIES="$VISION_MAX_RETRIES" \
    VISION_MAX_TOKENS="$VISION_MAX_TOKENS" \
    RULE_ONLY_JUDGE="$RULE_ONLY_JUDGE" \
        bash res-opd/scripts/eval_after_merge.sh "$BASE_MODEL_PATH" 0 "$VERSION_TAG" vision
else
    echo ""
    echo "[base] Skipped because RUN_BASE=${RUN_BASE}"
fi

echo ""
echo "[variants] Evaluating OSS checkpoints"
VLLM_BASE_PORT=$((VLLM_BASE_PORT + 1)) \
VISION_BENCHMARK="$VISION_BENCHMARK" \
VISION_BENCHMARK_DATA_DIR="$VISION_BENCHMARK_DATA_DIR" \
VISION_BENCHMARK_AUTO_DOWNLOAD="$VISION_BENCHMARK_AUTO_DOWNLOAD" \
VISION_BENCHMARK_CLEAN_SOURCE="$VISION_BENCHMARK_CLEAN_SOURCE" \
VISION_PARALLEL_WORKERS="$VISION_PARALLEL_WORKERS" \
VISION_MAX_RETRIES="$VISION_MAX_RETRIES" \
VISION_MAX_TOKENS="$VISION_MAX_TOKENS" \
RULE_ONLY_JUDGE="$RULE_ONLY_JUDGE" \
    bash res-opd/scripts/eval_batch_from_oss.sh \
        --oss-names "$TR075_RKL_OSS" "$TR075_SW_OSS" "$TR075_SW075_OSS" \
        --step "$STEP" \
        --eval-mode vision \
        --vision-benchmark "$VISION_BENCHMARK" \
        --vision-benchmark-data-dir "$VISION_BENCHMARK_DATA_DIR" \
        --vision-benchmark-auto-download "$VISION_BENCHMARK_AUTO_DOWNLOAD" \
        --vision-benchmark-clean-source "$VISION_BENCHMARK_CLEAN_SOURCE" \
        --rule-only-judge "$RULE_ONLY_JUDGE" \
        --version-tag "$VERSION_TAG"

echo ""
echo "Done."

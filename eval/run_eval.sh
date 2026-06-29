#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Vision-OPD Evaluation Script
#
# Supported benchmarks:
#   vstar, zoombench, hrbench-4k, hrbench-8k, mme-realworld, mme-realworld-cn, mme-realworld-lite,
#   mmstar, pope, pope_adv, pope_pop, pope_random, cv-bench, mmvp, visualprobe
#
# Usage:
#   API_BASE="http://localhost:8000/v1/" \
#   OPENAI_MODEL_ID="Vision-OPD-4B" \
#   BENCHMARK_DATA_DIR="/home/liuyanlin.lyl/notebook/data" \
#   BENCHMARK="vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn" \
#   bash eval/run_eval.sh
# =============================================================================

export MKL_SERVICE_FORCE_INTEL=1

# --- Required ---
API_BASE="${API_BASE:?ERROR: API_BASE must be set (e.g. http://localhost:8000/v1/)}"
OPENAI_MODEL_ID="${OPENAI_MODEL_ID:?ERROR: OPENAI_MODEL_ID must be set}"

# --- Optional ---
BENCHMARK="${BENCHMARK:-vstar}"
API_KEY="${OPENAI_API_KEY:-EMPTY}"
MODEL_NAME="${MODEL_NAME:-${OPENAI_MODEL_ID//\//_}}"
SEED="${SEED:-42}"
MAX_TOKENS_WAS_SET="${MAX_TOKENS+x}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
OUT_DIR="${OUT_DIR:-model_answer}"
JUDGE_DIR="${JUDGE_DIR:-judge}"
BENCHMARK_OUTPUT_LAYOUT="${BENCHMARK_OUTPUT_LAYOUT:-legacy}"
BENCHMARK_OUTPUT_SUFFIX="${BENCHMARK_OUTPUT_SUFFIX:-}"
MAX_RETRIES="${MAX_RETRIES:-3}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-256}"
ENABLE_THINKING="${ENABLE_THINKING:-}"
BENCHMARK_REFRESH_PROMPTS_WAS_SET="${BENCHMARK_REFRESH_PROMPTS+x}"
BENCHMARK_REFRESH_PROMPTS="${BENCHMARK_REFRESH_PROMPTS:-False}"

JUDGE_API_BASE="${JUDGE_API_BASE:-}"
JUDGE_API_KEY="${JUDGE_API_KEY:-}"
JUDGE_MODEL="${JUDGE_MODEL:-}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-}"
JUDGE_MAX_TOKENS="${JUDGE_MAX_TOKENS:-2048}"
RULE_ONLY_JUDGE_WAS_SET="${RULE_ONLY_JUDGE+x}"
RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-False}"
MCQ_EXTRACT_MODE_WAS_SET="${MCQ_EXTRACT_MODE+x}"
MCQ_EXTRACT_MODE="${MCQ_EXTRACT_MODE:-legacy}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-${SCRIPT_DIR}}"
BENCHMARK_PREPARE_DATA="${BENCHMARK_PREPARE_DATA:-True}"
BENCHMARK_AUTO_DOWNLOAD="${BENCHMARK_AUTO_DOWNLOAD:-True}"
BENCHMARK_CLEAN_SOURCE="${BENCHMARK_CLEAN_SOURCE:-False}"
mkdir -p "${BENCHMARK_DATA_DIR}"

if [[ ",${BENCHMARK}," == *",mmstar,"* || ",${BENCHMARK}," == *",cv-bench,"* ]]; then
  [[ -z "${BENCHMARK_REFRESH_PROMPTS_WAS_SET}" ]] && BENCHMARK_REFRESH_PROMPTS="True"
  [[ -z "${RULE_ONLY_JUDGE_WAS_SET}" ]] && RULE_ONLY_JUDGE="True"
  [[ -z "${MCQ_EXTRACT_MODE_WAS_SET}" ]] && MCQ_EXTRACT_MODE="official"
  [[ -z "${MAX_TOKENS_WAS_SET}" ]] && MAX_TOKENS="16"
fi

# Benchmark JSON mapping
declare -A BENCHMARK_JSON_MAP=(
  [zoombench]="zoombench.json"
  [vstar]="vstar.json"
  [hrbench-4k]="hr_bench_4k.json"
  [hrbench-8k]="hr_bench_8k.json"
  [mme-realworld]="MME_RealWorld.json"
  [mme-realworld-cn]="MME_RealWorld_CN.json"
  [mme-realworld-lite]="MME_RealWorld_Lite.json"
  [mmstar]="mmstar.json"
  [pope]="POPE.json"
  [pope_adv]="POPE_adv.json"
  [pope_pop]="POPE_pop.json"
  [pope_random]="POPE_random.json"
  [cv-bench]="cv_bench.json"
  [mmvp]="mmvp.json"
  [visualprobe]="visualprobe.json"
)

# Parse comma-separated benchmarks
IFS=',' read -r -a BENCHMARKS_TO_RUN <<< "${BENCHMARK}"

run_single_benchmark() {
  local bench="$1"
  local bench_json="${BENCHMARK_JSON_MAP[$bench]:-}"
  if [[ -z "${bench_json}" ]]; then
    echo "ERROR: Unsupported benchmark: ${bench}"
    exit 1
  fi

  echo "=========================================="
  echo "Benchmark: ${bench}"
  echo "API base: ${API_BASE}"
  echo "Model: ${OPENAI_MODEL_ID}"
  echo "Data dir: ${BENCHMARK_DATA_DIR}"
  echo "Prepare data: ${BENCHMARK_PREPARE_DATA}"
  echo "Auto download: ${BENCHMARK_AUTO_DOWNLOAD}"
  echo "Clean source: ${BENCHMARK_CLEAN_SOURCE}"
  echo "Refresh prompts: ${BENCHMARK_REFRESH_PROMPTS}"
  echo "Output suffix: ${BENCHMARK_OUTPUT_SUFFIX}"
  echo "=========================================="

  local model_tag="${MODEL_NAME}_seed${SEED}"
  local output_bench="${bench}${BENCHMARK_OUTPUT_SUFFIX}"
  local benchmark_json_path="${BENCHMARK_DATA_DIR}/${bench_json}"
  local out_dir_for_bench="${OUT_DIR}"
  local judge_dir_for_bench="${JUDGE_DIR}"
  local judge_json="${JUDGE_DIR}/${bench}/${model_tag}_answer.jsonl"
  local -a output_layout_args=()
  case "${BENCHMARK_OUTPUT_LAYOUT}" in
    legacy)
      ;;
    per_benchmark)
      out_dir_for_bench="${OUT_DIR}/${output_bench}/model_answer"
      judge_dir_for_bench="${JUDGE_DIR}/${output_bench}/judge"
      judge_json="${judge_dir_for_bench}/${model_tag}_answer.jsonl"
      output_layout_args+=(--no_benchmark_subdir)
      mkdir -p "${out_dir_for_bench}" "${judge_dir_for_bench}"
      ;;
    *)
      echo "ERROR: Unsupported BENCHMARK_OUTPUT_LAYOUT=${BENCHMARK_OUTPUT_LAYOUT}. Use legacy or per_benchmark." >&2
      exit 1
      ;;
  esac

  # [1/4] Prepare data
  echo "[1/4] Preparing data..."
  if [[ "${BENCHMARK_PREPARE_DATA}" == "False" || "${BENCHMARK_PREPARE_DATA}" == "false" || "${BENCHMARK_PREPARE_DATA}" == "0" ]]; then
    echo "Skipping data preparation; expecting prebuilt JSON/images."
  else
    if [[ ! -s "${benchmark_json_path}" ]]; then
      if [[ "${BENCHMARK_AUTO_DOWNLOAD}" == "False" || "${BENCHMARK_AUTO_DOWNLOAD}" == "false" || "${BENCHMARK_AUTO_DOWNLOAD}" == "0" ]]; then
        echo "ERROR: Benchmark JSON is missing and BENCHMARK_AUTO_DOWNLOAD is disabled: ${benchmark_json_path}" >&2
        exit 1
      fi
    fi
    local -a PREPARE_ARGS=(
      --benchmark "${bench}"
      --data_dir "${BENCHMARK_DATA_DIR}"
    )
    if [[ "${BENCHMARK_REFRESH_PROMPTS}" == "True" || "${BENCHMARK_REFRESH_PROMPTS}" == "true" || "${BENCHMARK_REFRESH_PROMPTS}" == "1" ]]; then
      PREPARE_ARGS+=(--refresh-prompts)
    fi
    if [[ "${BENCHMARK_CLEAN_SOURCE}" == "True" || "${BENCHMARK_CLEAN_SOURCE}" == "true" || "${BENCHMARK_CLEAN_SOURCE}" == "1" ]]; then
      PREPARE_ARGS+=(--clean-source)
    fi
    "$PYTHON_BIN" prepare_data.py "${PREPARE_ARGS[@]}"
  fi
  if [[ ! -s "${benchmark_json_path}" ]]; then
    echo "ERROR: Prepared benchmark JSON is missing or empty: ${benchmark_json_path}" >&2
    exit 1
  fi

  # [2/4] Inference
  echo "[2/4] Running inference..."
  local -a INFER_ARGS=(
    --benchmark "${bench}"
    --benchmark_json "${benchmark_json_path}"
    --out_dir "${out_dir_for_bench}"
    --model_name "${model_tag}"
    --seed "${SEED}"
    --api_base "${API_BASE}"
    --api_key "${API_KEY}"
    --model_id "${OPENAI_MODEL_ID}"
    --max_tokens "${MAX_TOKENS}"
    --max_retries "${MAX_RETRIES}"
    --parallel_workers "${PARALLEL_WORKERS}"
  )
  [[ -n "${ENABLE_THINKING}" ]] && INFER_ARGS+=(--enable_thinking "${ENABLE_THINKING}")
  INFER_ARGS+=("${output_layout_args[@]}")

  "$PYTHON_BIN" infer.py "${INFER_ARGS[@]}"

  # [3/4] Judge
  echo "[3/4] Running judge..."
  local -a JUDGE_ARGS=()
  [[ -n "${JUDGE_API_BASE}" ]] && JUDGE_ARGS+=(--api_base "${JUDGE_API_BASE}")
  [[ -n "${JUDGE_API_KEY}" ]] && JUDGE_ARGS+=(--api_key "${JUDGE_API_KEY}")
  [[ -n "${JUDGE_MODEL}" ]] && JUDGE_ARGS+=(--judge_model "${JUDGE_MODEL}")
  [[ -n "${JUDGE_MODEL_PATH}" ]] && JUDGE_ARGS+=(--judge_model_path "${JUDGE_MODEL_PATH}")
  [[ -n "${JUDGE_MAX_TOKENS}" ]] && JUDGE_ARGS+=(--judge_max_tokens "${JUDGE_MAX_TOKENS}")
  [[ "${RULE_ONLY_JUDGE}" == "True" || "${RULE_ONLY_JUDGE}" == "true" ]] && JUDGE_ARGS+=(--rule_only)

  local judge_model_tag="${model_tag}"

  "$PYTHON_BIN" judge_qwenlm.py \
    --benchmark "${bench}" \
    --model "${judge_model_tag}" \
    --answer_dir "${out_dir_for_bench}" \
    --judge_dir "${judge_dir_for_bench}" \
    --mcq_extract_mode "${MCQ_EXTRACT_MODE}" \
    "${output_layout_args[@]}" \
    "${JUDGE_ARGS[@]}"

  # [4/4] Accuracy
  echo "[4/4] Calculating accuracy..."
  "$PYTHON_BIN" cal_acc.py \
    --benchmark "${bench}" \
    --judge_json "${judge_json}" \
    --benchmark_json "${benchmark_json_path}"

  echo "Done: ${bench}"
}

# Main loop
echo "Benchmarks to run: ${BENCHMARKS_TO_RUN[*]}"
total="${#BENCHMARKS_TO_RUN[@]}"
for idx in "${!BENCHMARKS_TO_RUN[@]}"; do
  bench="${BENCHMARKS_TO_RUN[$idx]}"
  bench="$(echo "${bench}" | xargs)"  # trim spaces
  [[ -z "${bench}" ]] && continue
  echo
  echo "########## [$((idx + 1))/${total}] ${bench} ##########"
  run_single_benchmark "${bench}"
done

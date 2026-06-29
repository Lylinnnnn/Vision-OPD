#!/usr/bin/env bash
set -euo pipefail

# Clean MMStar rerun for base + tr0.75 variants.
# This uses a letter-only MMStar prompt and strict deterministic MCQ extraction,
# and writes under each checkpoint's normal eval_results/latest folder.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

export VISION_BENCHMARK="${VISION_BENCHMARK:-mmstar}"
export VERSION_TAG="${VERSION_TAG:-latest}"
export VISION_BENCHMARK_REFRESH_PROMPTS="${VISION_BENCHMARK_REFRESH_PROMPTS:-True}"
export VISION_BENCHMARK_OUTPUT_SUFFIX="${VISION_BENCHMARK_OUTPUT_SUFFIX:-}"
export RULE_ONLY_JUDGE="${RULE_ONLY_JUDGE:-True}"
export MCQ_EXTRACT_MODE="${MCQ_EXTRACT_MODE:-official}"
export VISION_MAX_TOKENS="${VISION_MAX_TOKENS:-16}"
export VISION_PARALLEL_WORKERS="${VISION_PARALLEL_WORKERS:-128}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-8030}"
export VLLM_PORT_CLEANUP="${VLLM_PORT_CLEANUP:-True}"
export FORCE_VISION_EVAL="${FORCE_VISION_EVAL:-True}"

exec bash "$SCRIPT_DIR/eval_mmstar_cvbench_base_tr075.sh"

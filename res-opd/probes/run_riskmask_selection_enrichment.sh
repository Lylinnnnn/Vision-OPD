#!/usr/bin/env bash
# Offline RiskMask top-p enrichment analysis over same-image KL probe traces.
#
# Required:
#   TRACE_JSONL=/path/to/pair_kl_trace.jsonl
# or
#   TRACE_GLOB="/path/to/shards/*.jsonl"
#
# Optional:
#   OUTPUT_DIR, TOP_P, RANK_SCOPE, RANDOM_SEED, MENTION_TOKEN_THRESHOLD, PYTHON_BIN

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    /home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 \
    /home/zhengyanzhao.zyz/.conda/envs/vision-opd/bin/python3
  do
    if [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

OUTPUT_DIR="${OUTPUT_DIR:-${RES_OPD_ROOT}/probes/results/riskmask_selection_enrichment}"
TOP_P="${TOP_P:-0.30}"
RANK_SCOPE="${RANK_SCOPE:-global}"
RANDOM_SEED="${RANDOM_SEED:-42}"
MENTION_TOKEN_THRESHOLD="${MENTION_TOKEN_THRESHOLD:-0.0}"

ARGS=(
  --output-dir "$OUTPUT_DIR"
  --top-p "$TOP_P"
  --rank-scope "$RANK_SCOPE"
  --random-seed "$RANDOM_SEED"
  --mention-token-threshold "$MENTION_TOKEN_THRESHOLD"
)

if [[ -n "${TRACE_JSONL:-}" ]]; then
  IFS=',' read -r -a TRACE_LIST <<< "$TRACE_JSONL"
  for trace in "${TRACE_LIST[@]}"; do
    if [[ -n "$trace" ]]; then
      ARGS+=(--trace-jsonl "$trace")
    fi
  done
fi

if [[ -n "${TRACE_GLOB:-}" ]]; then
  IFS=',' read -r -a TRACE_GLOB_LIST <<< "$TRACE_GLOB"
  for trace_glob in "${TRACE_GLOB_LIST[@]}"; do
    if [[ -n "$trace_glob" ]]; then
      ARGS+=(--trace-glob "$trace_glob")
    fi
  done
fi

echo "============================================================"
echo " RiskMask Selection Enrichment"
echo "============================================================"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "TRACE_JSONL = ${TRACE_JSONL:-}"
echo "TRACE_GLOB = ${TRACE_GLOB:-}"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "TOP_P = $TOP_P"
echo "RANK_SCOPE = $RANK_SCOPE"
echo "RANDOM_SEED = $RANDOM_SEED"
echo "MENTION_TOKEN_THRESHOLD = $MENTION_TOKEN_THRESHOLD"
echo "============================================================"

exec "$PYTHON_BIN" "${SCRIPT_DIR}/analyze_riskmask_selection_enrichment.py" "${ARGS[@]}"

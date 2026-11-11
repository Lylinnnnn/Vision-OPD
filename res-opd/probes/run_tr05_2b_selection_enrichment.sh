#!/usr/bin/env bash
# Reuse existing 2B captions, score original vs tr=0.5, and rebuild selection table.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"
: "${BASE_RESULTS:?Set BASE_RESULTS to Base eval_results.jsonl}"
: "${RANDOM_RESULTS:?Set RANDOM_RESULTS to RandomMask step156 eval_results.jsonl}"
: "${RISK_RESULTS:?Set RISK_RESULTS to RiskMask step150 eval_results.jsonl}"

"$PYTHON_BIN" res-opd/probes/preflight_tr05_2b_enrichment.py \
  --base-results "$BASE_RESULTS" \
  --random-results "$RANDOM_RESULTS" \
  --risk-results "$RISK_RESULTS"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/res-opd/probes/results/selection_enrichment_2b_tr05_official_v2}"
MODEL_PATH="${MODEL_PATH:-/home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct}" \
EVAL_RESULTS="$BASE_RESULTS" \
IMAGE_ROOT="${IMAGE_ROOT:-/data/oss_bucket_0/yanlin/COCO/coco2017val/val2017}" \
OUTPUT_DIR="$OUTPUT_DIR" \
TEACHER_RATIO=0.5 STUDENT_RATIO=1.0 TOP_P=0.30 RANK_SCOPE=global \
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}" \
PYTHON_BIN="$PYTHON_BIN" \
bash res-opd/probes/run_base_dual_view_kl_probe.sh

"$PYTHON_BIN" -c 'import json,sys; p=sys.argv[1]; data=json.load(open(p)); n=data["num_records"]; assert n == 1000, "Incomplete trace: %s/1000" % n; print("OFFICIAL_SELECTION_TABLE=PASS", p)' \
  "$OUTPUT_DIR/riskmask_selection_enrichment_summary.json"

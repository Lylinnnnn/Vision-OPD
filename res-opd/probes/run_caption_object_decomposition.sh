#!/usr/bin/env bash
# Offline decomposition of CHAIR-style object changes across Res-OPD variants.
#
# Typical server launch from repo root:
#
#   tmux new-session -d -s caption_decomp \
#     "bash res-opd/probes/run_caption_object_decomposition.sh 2>&1 | tee res-opd/logs/caption_object_decomposition.log"
#
# Optional overrides:
#   BASE_RESULTS, RUN_SPECS, RUNS_JSON, OUTPUT_DIR, TEST_JSON, PYTHON_BIN, ALLOW_MISSING
#
# RUN_SPECS format:
#   name=/path/eval_results.jsonl,name2=/path2/eval_results.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RES_OPD_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VISION_OPD_ROOT="$(cd "${RES_OPD_ROOT}/.." && pwd)"

cd "$VISION_OPD_ROOT"

export PYTHONPATH="$VISION_OPD_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"

# Match the training/probe launchers: prefer conda-bundled cuDNN over system /lib64.
CONDA_CUDNN_LIB="$("$PYTHON_BIN" -c "
import os, sys
try:
    import nvidia.cudnn
    p = getattr(nvidia.cudnn, '__file__', None)
    if p:
        d = os.path.join(os.path.dirname(p), 'lib')
        if os.path.isdir(d):
            print(d); sys.exit(0)
except Exception:
    pass
try:
    import nvidia
    d = os.path.join(nvidia.__path__[0], 'cudnn', 'lib')
    if os.path.isdir(d):
        print(d); sys.exit(0)
except Exception:
    pass
" 2>/dev/null || true)"
if [[ -z "$CONDA_CUDNN_LIB" || ! -d "$CONDA_CUDNN_LIB" ]]; then
  _FALLBACK="$(dirname "$PYTHON_BIN")/../lib/python3.12/site-packages/nvidia/cudnn/lib"
  if [[ -d "$_FALLBACK" ]]; then
    CONDA_CUDNN_LIB="$_FALLBACK"
  fi
fi
if [[ -n "$CONDA_CUDNN_LIB" && -d "$CONDA_CUDNN_LIB" ]]; then
  export LD_LIBRARY_PATH="${CONDA_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

BASE_DIR="/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/latest/full"
BASE_RESULTS="${BASE_RESULTS:-${BASE_DIR}/Qwen3VL-2B-Instruct/train5000_test1000_original_sr1p0/eval_results.jsonl}"
TEST_JSON="${TEST_JSON:-${RES_OPD_ROOT}/data/test_1000.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${RES_OPD_ROOT}/probes/results/caption_object_decomposition/full5k_base_vs_variants}"
TOP_EXAMPLES="${TOP_EXAMPLES:-25}"
ALLOW_MISSING="${ALLOW_MISSING:-False}"

DEFAULT_RUN_SPECS="tr10_rkl=${BASE_DIR}/Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr1.0-a1.0-frozen-rkl-full5k-e1_global_step_39/train5000_test1000_original_sr1p0/eval_results.jsonl"
DEFAULT_RUN_SPECS+=",tr075_rkl=${BASE_DIR}/Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-full5k-e1_global_step_39/train5000_test1000_original_sr1p0/eval_results.jsonl"
DEFAULT_RUN_SPECS+=",tr075_rkl_sw=${BASE_DIR}/Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw-full5k-e1_global_step_39/train5000_test1000_original_sr1p0/eval_results.jsonl"
DEFAULT_RUN_SPECS+=",tr075_rkl_sw075=${BASE_DIR}/Res-OPD-Qwen3VL-2B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-sw075-full5k-e1_global_step_39/train5000_test1000_original_sr1p0/eval_results.jsonl"
RUN_SPECS="${RUN_SPECS:-$DEFAULT_RUN_SPECS}"

mkdir -p "${RES_OPD_ROOT}/logs" "$OUTPUT_DIR"

ARGS=(
  --base-results "$BASE_RESULTS"
  --test-json "$TEST_JSON"
  --output-dir "$OUTPUT_DIR"
  --top-examples "$TOP_EXAMPLES"
)

if [[ -n "${RUNS_JSON:-}" ]]; then
  ARGS+=(--runs-json "$RUNS_JSON")
else
  IFS=',' read -r -a RUN_SPEC_LIST <<< "$RUN_SPECS"
  for spec in "${RUN_SPEC_LIST[@]}"; do
    if [[ -n "$spec" ]]; then
      ARGS+=(--run "$spec")
    fi
  done
fi

case "$ALLOW_MISSING" in
  True|true|TRUE|1|yes|YES|y|Y)
    ARGS+=(--allow-missing)
    ;;
esac

echo "============================================================"
echo " Caption Object Change Decomposition"
echo "============================================================"
echo "PYTHON_BIN = $PYTHON_BIN"
echo "BASE_RESULTS = $BASE_RESULTS"
echo "TEST_JSON = $TEST_JSON"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "TOP_EXAMPLES = $TOP_EXAMPLES"
echo "ALLOW_MISSING = $ALLOW_MISSING"
if [[ -n "${RUNS_JSON:-}" ]]; then
  echo "RUNS_JSON = $RUNS_JSON"
else
  echo "RUN_SPECS = $RUN_SPECS"
fi
echo "============================================================"

exec "$PYTHON_BIN" "${SCRIPT_DIR}/decompose_caption_object_changes.py" "${ARGS[@]}"

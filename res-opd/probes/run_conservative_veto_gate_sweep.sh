#!/usr/bin/env bash
# Offline conservative-veto gate sweep on an existing pair_kl_trace.jsonl.
#
# Typical server launch from repo root:
#
#   tmux new-session -d -s veto_gate_sweep \
#     "bash res-opd/probes/run_conservative_veto_gate_sweep.sh 2>&1 | tee res-opd/logs/veto_gate_sweep.log"
#
# Optional overrides:
#   TRACE_JSONL, OUTPUT_ROOT, OUTPUT_DIR, PYTHON_BIN, MAX_EXAMPLES_PER_GATE
#
# By default, different MAX_EXAMPLES_PER_GATE values write to separate
# directories under OUTPUT_ROOT, e.g. max_examples_20 and max_examples_100.
# Set OUTPUT_DIR explicitly only when you intentionally want a fixed path.

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

TRACE_JSONL="${TRACE_JSONL:-/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/same_image_rkl_signal/dual_view_base_sr10_full_vs_lowres075/pair_kl_trace.jsonl}"
MAX_EXAMPLES_PER_GATE="${MAX_EXAMPLES_PER_GATE:-20}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RES_OPD_ROOT}/probes/results/conservative_veto_gate_sweep/dual_view_base_sr10_full_vs_lowres075}"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
  OUTPUT_DIR="${OUTPUT_ROOT}/max_examples_${MAX_EXAMPLES_PER_GATE}"
fi

mkdir -p "${RES_OPD_ROOT}/logs" "$OUTPUT_DIR"

if [[ ! -s "$TRACE_JSONL" ]]; then
  echo "ERROR: TRACE_JSONL is missing or empty: $TRACE_JSONL" >&2
  echo "Set TRACE_JSONL=/path/to/pair_kl_trace.jsonl if your result path is different." >&2
  exit 1
fi

echo "============================================================"
echo " Conservative Veto Gate Sweep"
echo "============================================================"
echo "PYTHON_BIN            = $PYTHON_BIN"
echo "TRACE_JSONL           = $TRACE_JSONL"
echo "OUTPUT_ROOT           = $OUTPUT_ROOT"
echo "OUTPUT_DIR            = $OUTPUT_DIR"
echo "MAX_EXAMPLES_PER_GATE = $MAX_EXAMPLES_PER_GATE"
echo "============================================================"

exec "$PYTHON_BIN" "${SCRIPT_DIR}/sweep_conservative_veto_gates.py" \
  --trace-jsonl "$TRACE_JSONL" \
  --output-dir "$OUTPUT_DIR" \
  --max-examples-per-gate "$MAX_EXAMPLES_PER_GATE"

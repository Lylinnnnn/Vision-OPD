#!/usr/bin/env bash
set -euo pipefail

# Download Qwen3-VL Thinking checkpoints from ModelScope.
#
# Usage:
#   bash res-opd/scripts/tmp/download_qwen3vl_thinking_modelscope.sh
#
# Optional:
#   MODEL_ROOT=/home/liuyanlin.lyl/notebook/model/qwen \
#   MODELS="Qwen/Qwen3-VL-2B-Thinking Qwen/Qwen3-VL-8B-Thinking" \
#   FORCE_DOWNLOAD=False \
#   bash res-opd/scripts/tmp/download_qwen3vl_thinking_modelscope.sh

MODEL_ROOT="${MODEL_ROOT:-/home/liuyanlin.lyl/notebook/model/qwen}"
MODELS="${MODELS:-Qwen/Qwen3-VL-2B-Thinking Qwen/Qwen3-VL-8B-Thinking}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-False}"
PYTHON_BIN="${PYTHON_BIN:-/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3}"

mkdir -p "$MODEL_ROOT"

"$PYTHON_BIN" - "$MODEL_ROOT" "$MODELS" "$FORCE_DOWNLOAD" <<'PY'
import os
import shutil
import sys
from pathlib import Path

model_root = Path(sys.argv[1]).expanduser().resolve()
models = [m.strip() for m in sys.argv[2].split() if m.strip()]
force = sys.argv[3].lower() in {"1", "true", "yes", "y"}

try:
    from modelscope import snapshot_download
except Exception as exc:
    raise SystemExit(
        "ModelScope is not importable in this Python environment. "
        "Install it in the server env first, e.g. `pip install modelscope`. "
        f"Original error: {exc}"
    )

for model_id in models:
    local_name = model_id.rstrip("/").split("/")[-1]
    target = model_root / local_name
    config_path = target / "config.json"

    if config_path.exists() and not force:
        print(f"[skip] {model_id} already exists at {target}")
        continue

    if target.exists() and force:
        print(f"[remove] {target}")
        shutil.rmtree(target)

    print(f"[download] {model_id} -> {target}")
    snapshot_download(model_id, local_dir=str(target))
    if not config_path.exists():
        raise SystemExit(f"Download finished but config.json is missing: {config_path}")
    print(f"[done] {target}")
PY

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:?Activate vision-opd first}/bin/python3}"
GPU_LIST="${GPU_LIST:-1,3,5,7}"
BASE_PORT="${BASE_PORT:-9200}"
MANIFEST="${ROOT}/res-opd/eval/manifests/teacher_scale_table5_chair_missing_6.txt"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python executable not found: $PYTHON_BIN" >&2
    exit 1
fi

while IFS='|' read -r index label source_type primary oss_name step student_ratio; do
    if [[ -z "$index" || "$index" == \#* ]]; then
        continue
    fi
    target="${ROOT}/res-opd/checkpoints/${primary}/${step}"
    if [[ -e "$target" || -L "$target" ]]; then
        echo "Refusing download-mode evaluation: local checkpoint already exists: $target" >&2
        exit 1
    fi
done < "$MANIFEST"

cd "$ROOT"
exec "$PYTHON_BIN" res-opd/scripts/tmp/run_paper_chair_official_batch.py \
    --manifest "$MANIFEST" --indices 1-6 \
    --base-port "$BASE_PORT" --gpu-list "$GPU_LIST" --eval-shard-count 4 \
    --prompt 'Describe the image.' --max-samples 1000 \
    --output-suffix chair_teacher_scale_table5_official_v2 \
    --reuse-output-suffix chair_official_v2 \
    --reuse-output-suffix chair_describe_image_official \
    --reuse-output-suffix three_tables_fill_official_v2 \
    --oss-checkpoint-access download \
    --transition-wait-seconds 300 --require-idle-gpus "$@"

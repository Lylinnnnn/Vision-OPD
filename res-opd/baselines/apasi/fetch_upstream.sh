#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/../.." && pwd)/third_party/baselines/APASI"
PIN=849fa4c212e6fc97c7b56abdfe65d20b1d73292b
if [[ ! -e "$DEST" ]]; then
    git clone --depth 1 https://github.com/davidluciolu/APASI.git "$DEST"
    if [[ "$(git -C "$DEST" rev-parse HEAD)" != "$PIN" ]]; then
        git -C "$DEST" fetch --depth 1 origin "$PIN"
        git -C "$DEST" checkout --detach "$PIN"
    fi
fi
test -d "$DEST/.git" || { echo "Unexpected APASI source path: $DEST" >&2; exit 1; }
ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
test "$ACTUAL" = "$PIN" || {
    echo "APASI source commit changed: expected $PIN, got $ACTUAL" >&2
    exit 1
}
test -s "$DEST/llava/prepare_data/co_graph_construction.py"
test -s "$DEST/playground/data/neg_data/LVIS_CATEGORIES.json"
echo "APASI_UPSTREAM=PASS commit=$ACTUAL path=$DEST"

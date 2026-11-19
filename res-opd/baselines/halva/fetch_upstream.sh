#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEST="$(cd "$SCRIPT_DIR/../.." && pwd)/third_party/baselines/HALVA"
PIN=059b086f7b4d3d5af436f043d5c9eb9cc627c3e0
if [[ -e "$DEST" ]]; then
    test -d "$DEST/.git" || { echo "Unexpected existing path: $DEST" >&2; exit 1; }
else
    git clone --depth 1 https://github.com/pritamqu/HALVA.git "$DEST"
fi
ACTUAL="$(git -C "$DEST" rev-parse HEAD)"
test "$ACTUAL" = "$PIN" || {
    echo "HALVA source commit changed: expected $PIN, got $ACTUAL" >&2
    exit 1
}
echo "HALVA_UPSTREAM=PASS commit=$ACTUAL path=$DEST"

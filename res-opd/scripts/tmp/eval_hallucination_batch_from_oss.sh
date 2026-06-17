#!/bin/bash

set -euo pipefail

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Temporary final-hallucination benchmark runner.
# Keep this under scripts/tmp/ while AMBER/MME wiring is being tested.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Batch AMBER / classic MME perception evaluation from OSS checkpoints.
#
# Downloads merged checkpoints from OSS, runs selected final hallucination
# benchmarks, deletes model files, and keeps only eval_results.
#
# Usage:
#   AMBER_ROOT=/path/to/AMBER AMBER_IMAGE_ROOT=/path/to/AMBER/images \
#   MME_ROOT=/path/to/MME_Benchmark_release_version \
#   bash res-opd/scripts/tmp/eval_hallucination_batch_from_oss.sh \
#       --oss-names <name1> [name2] ... \
#       [--step global_step_40] [--version-tag final_hallu] \
#       [--benchmarks amber,mme]
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
CKPT_BASE="${RES_OPD_ROOT}/checkpoints"
AMBER_SCRIPT="${RES_OPD_ROOT}/scripts/tmp/val_amber.sh"
MME_SCRIPT="${RES_OPD_ROOT}/scripts/tmp/val_mme_perception.sh"

STEP="global_step_40"
VERSION_TAG="final_hallucination"
BENCHMARKS="amber,mme"
OSS_NAMES=()
LOCAL_NAMES=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --oss-names)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                OSS_NAMES+=("$1")
                shift
            done
            ;;
        --local-names)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                LOCAL_NAMES+=("$1")
                shift
            done
            ;;
        --step) STEP="$2"; shift 2 ;;
        --version-tag) VERSION_TAG="$2"; shift 2 ;;
        --benchmarks) BENCHMARKS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#OSS_NAMES[@]} -eq 0 ]]; then
    echo "Error: --oss-names is required" >&2
    exit 1
fi

has_benchmark() {
    local name="$1"
    [[ ",${BENCHMARKS}," == *",${name},"* ]]
}

derive_local_name() {
    local oss_name="$1"
    if [[ "$oss_name" =~ ^ResOPD_s([0-9]+)_t([0-9]+)_a([0-9.]+)_(.+)$ ]]; then
        echo "Res-OPD-Qwen3VL-2B-Instruct-s${BASH_REMATCH[1]}-t${BASH_REMATCH[2]}-a${BASH_REMATCH[3]}-${BASH_REMATCH[4]}"
    elif [[ "$oss_name" =~ ^ResOPD_s([0-9]+)_a([0-9.]+)_(.+)$ ]]; then
        echo "Res-OPD-Qwen3VL-2B-Instruct-s${BASH_REMATCH[1]}-a${BASH_REMATCH[2]}-${BASH_REMATCH[3]}"
    else
        echo "$oss_name" | sed \
            -e 's/^ResOPD_/Res-OPD-Qwen3VL-2B-Instruct-/' \
            -e 's/_a/-a/' \
            -e 's/_t/-t/g' \
            -e 's/_s/-s/' \
            -e 's/_/-/g'
    fi
}

download_checkpoint() {
    local oss_path="$1"
    local local_ckpt_dir="$2"
    mkdir -p "${local_ckpt_dir}/actor/huggingface"
    ossutil cp "${oss_path}/model.safetensors" "${local_ckpt_dir}/model.safetensors" -f
    for f in config.json tokenizer_config.json tokenizer.json chat_template.jinja generation_config.json processor_config.json; do
        ossutil cp "${oss_path}/${f}" "${local_ckpt_dir}/actor/huggingface/${f}" -f 2>/dev/null || true
    done
}

cleanup_checkpoint() {
    local local_ckpt_dir="$1"
    rm -f "${local_ckpt_dir}/model.safetensors"
    rm -rf "${local_ckpt_dir}/actor"
}

echo "=========================================="
echo " Final Hallucination Batch Eval from OSS"
echo " Step:        ${STEP}"
echo " Version tag: ${VERSION_TAG}"
echo " Benchmarks:  ${BENCHMARKS}"
echo " Experiments: ${#OSS_NAMES[@]}"
echo " Started at:  $(date)"
echo "=========================================="

for idx in "${!OSS_NAMES[@]}"; do
    oss_name="${OSS_NAMES[$idx]}"
    if [[ $idx -lt ${#LOCAL_NAMES[@]} && -n "${LOCAL_NAMES[$idx]}" ]]; then
        local_exp_name="${LOCAL_NAMES[$idx]}"
    else
        local_exp_name="$(derive_local_name "$oss_name")"
    fi

    oss_path="${OSS_BASE}/${oss_name}/${STEP}"
    local_ckpt_dir="${CKPT_BASE}/${local_exp_name}/${STEP}"

    echo ""
    echo "=========================================="
    echo " Processing: ${oss_name}"
    echo " Local exp:  ${local_exp_name}"
    echo " OSS:        ${oss_path}"
    echo "=========================================="

    echo "[1/4] Downloading checkpoint ..."
    download_checkpoint "$oss_path" "$local_ckpt_dir"

    if has_benchmark amber; then
        echo "[2/4] Running AMBER ..."
        bash "$AMBER_SCRIPT" "$local_ckpt_dir" "$VERSION_TAG"
    fi

    if has_benchmark mme; then
        echo "[3/4] Running classic MME perception ..."
        bash "$MME_SCRIPT" "$local_ckpt_dir" "$VERSION_TAG"
    fi

    echo "[4/4] Cleaning up model files ..."
    cleanup_checkpoint "$local_ckpt_dir"

    echo "Done: ${oss_name}"
done

echo ""
echo "=========================================="
echo " All final hallucination evals completed."
echo " Finished at: $(date)"
echo "=========================================="

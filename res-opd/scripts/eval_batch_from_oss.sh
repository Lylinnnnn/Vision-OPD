#!/bin/bash
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ⚠️  Log 规则: 批量 eval 无单独 log 文件，结果存储在
#     eval_results/<version_tag>/<experiment_name>_<step>/<dataset>/
#     禁止在 logs/ 下创建 eval 相关子目录！
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Batch Evaluation from OSS (Parameterized)
# Downloads merged checkpoints from OSS, runs eval, deletes model files,
# keeps only eval results.
#
# Usage:
#   bash scripts/eval_batch_from_oss.sh --oss-names <name1> [name2] ... \
#       [--step global_step_92] [--student-px 448] [--version-tag v5]
#
# Examples:
#   # Evaluate specific experiments at step 92
#   bash scripts/eval_batch_from_oss.sh \
#       --oss-names ResOPD_s448_t0_a0.5_ema-e2 ResOPD_s448_t448_a0.5_ema-e2 \
#       --step global_step_92 --student-px 448 --version-tag v5
#
#   # Evaluate all ema experiments at default step
#   bash scripts/eval_batch_from_oss.sh \
#       --oss-names ResOPD_s448_t200_a0.5_ema-e2 \
#       --step global_step_92
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

OSS_BASE="oss://industry-algo/yanlin/ckpt/OPD/v4"
CKPT_BASE="${RES_OPD_ROOT}/checkpoints"
EVAL_SCRIPT="${RES_OPD_ROOT}/scripts/eval_after_merge.sh"

# Defaults
STEP="global_step_92"
STUDENT_PX=448
VERSION_TAG="latest"
OSS_NAMES=()
LOCAL_NAMES=()

# Parse arguments
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
        --step)       STEP="$2"; shift 2 ;;
        --student-px) STUDENT_PX="$2"; shift 2 ;;
        --version-tag) VERSION_TAG="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#OSS_NAMES[@]} -eq 0 ]]; then
    echo "Error: --oss-names is required" >&2
    echo "Usage: $0 --oss-names <name1> [name2] ... [--step STEP] [--student-px PX] [--version-tag TAG]" >&2
    exit 1
fi

echo "=========================================="
echo " Batch Eval from OSS"
echo " Step: ${STEP}"
echo " Student PX: ${STUDENT_PX}"
echo " Version tag: ${VERSION_TAG}"
echo " Experiments: ${#OSS_NAMES[@]}"
echo " Started at: $(date)"
echo "=========================================="

for idx in "${!OSS_NAMES[@]}"; do
    oss_name="${OSS_NAMES[$idx]}"

    # Derive local experiment name with triple fallback:
    #   1. Explicit --local-names (if provided for this index)
    #   2. Structured regex parsing
    #   3. sed fallback (legacy compatibility)
    if [[ $idx -lt ${#LOCAL_NAMES[@]} && -n "${LOCAL_NAMES[$idx]}" ]]; then
        local_exp_name="${LOCAL_NAMES[$idx]}"
    elif [[ "$oss_name" =~ ^ResOPD_s([0-9]+)_t([0-9]+)_a([0-9.]+)_(.+)$ ]]; then
        local_exp_name="Res-OPD-Qwen3VL-2B-Instruct-s${BASH_REMATCH[1]}-t${BASH_REMATCH[2]}-a${BASH_REMATCH[3]}-${BASH_REMATCH[4]}"
    elif [[ "$oss_name" =~ ^ResOPD_s([0-9]+)_a([0-9.]+)_(.+)$ ]]; then
        local_exp_name="Res-OPD-Qwen3VL-2B-Instruct-s${BASH_REMATCH[1]}-a${BASH_REMATCH[2]}-${BASH_REMATCH[3]}"
    else
        # sed fallback for legacy/non-standard names
        local_exp_name=$(echo "$oss_name" | sed \
            -e 's/^ResOPD_/Res-OPD-Qwen3VL-2B-Instruct-/' \
            -e 's/_a/-a/' \
            -e 's/_t/-t/g' \
            -e 's/_s/-s/' \
            -e 's/_/-/g')
        echo "⚠️  Regex failed for '${oss_name}', using sed fallback: ${local_exp_name}"
    fi

    oss_path="${OSS_BASE}/${oss_name}/${STEP}"
    local_ckpt_dir="${CKPT_BASE}/${local_exp_name}/${STEP}"

    echo ""
    echo "=========================================="
    echo " Processing: ${oss_name}"
    echo " Local exp: ${local_exp_name}"
    echo " OSS: ${oss_path}"
    echo "=========================================="

    # Check if eval results already exist
    if ls "${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${local_exp_name}_${STEP}/"*/chair_metrics.json 2>/dev/null | grep -q .; then
        echo "⚠️  Eval results already exist, skipping."
        continue
    fi

    # Step 1: Download checkpoint from OSS
    echo "[1/4] Downloading from OSS ..."
    mkdir -p "${local_ckpt_dir}/actor/huggingface"
    ossutil cp "${oss_path}/model.safetensors" "${local_ckpt_dir}/model.safetensors" -f
    for f in config.json tokenizer_config.json tokenizer.json chat_template.jinja generation_config.json processor_config.json; do
        ossutil cp "${oss_path}/${f}" "${local_ckpt_dir}/actor/huggingface/${f}" -f 2>/dev/null || true
    done
    echo "  ✅ Download complete"

    # Step 2: Run evaluation
    echo "[2/4] Running evaluation ..."
    bash "$EVAL_SCRIPT" "$local_ckpt_dir" "$STUDENT_PX" "$VERSION_TAG"
    echo "  ✅ Evaluation complete"

    # Step 3: Delete model files (keep eval results)
    echo "[3/4] Cleaning up model files ..."
    rm -f "${local_ckpt_dir}/model.safetensors"
    rm -rf "${local_ckpt_dir}/actor"
    echo "  ✅ Model files deleted"

    # Step 4: Verify eval results
    echo "[4/4] Verifying eval results ..."
    result_dir="${RES_OPD_ROOT}/eval_results/${VERSION_TAG}/${local_exp_name}_${STEP}"
    if ls "${result_dir}/"*/chair_metrics.json 2>/dev/null | grep -q .; then
        echo "  ✅ Eval results saved:"
        find "${result_dir}" -name "chair_metrics.json" -exec echo "    {}" \;
    else
        echo "  ⚠️  No eval results found!"
    fi

    echo ""
    echo "✅ Done: ${oss_name}"
done

echo ""
echo "=========================================="
echo " All evaluations completed!"
echo " Finished at: $(date)"
echo "=========================================="

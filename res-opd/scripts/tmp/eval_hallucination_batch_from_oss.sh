#!/bin/bash

set -euo pipefail

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Temporary final-hallucination benchmark runner.
# Keep this under scripts/tmp/ while AMBER/MME wiring is being tested.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# =============================================================================
# Batch COCO / AMBER / classic MME perception evaluation from OSS checkpoints.
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
#       [--benchmarks chair,pope,amber,mme] \
#       [--student-px 448] [--target-px 448] \
#       [--degradation-mode square] [--student-ratio 1.0]
#
# Benchmark aliases:
#   coco  -> chair,pope
#   final -> amber,mme
#   all   -> chair,pope,amber,mme
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RES_OPD_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VISION_OPD_ROOT="$(cd "$RES_OPD_ROOT/.." && pwd)"
cd "$VISION_OPD_ROOT"

OSS_BASE="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}"
CKPT_BASE="${RES_OPD_ROOT}/checkpoints"
COCO_SCRIPT="${RES_OPD_ROOT}/scripts/eval_after_merge.sh"
AMBER_SCRIPT="${RES_OPD_ROOT}/scripts/tmp/val_amber.sh"
MME_SCRIPT="${RES_OPD_ROOT}/scripts/tmp/val_mme_perception.sh"
HF_FILES=(
    config.json
    tokenizer_config.json
    tokenizer.json
    chat_template.jinja
    generation_config.json
    processor_config.json
    preprocessor_config.json
    image_processor_config.json
    video_processor_config.json
    special_tokens_map.json
    tokenizer.model
    merges.txt
    vocab.json
    model.safetensors.index.json
)

STEP="global_step_40"
VERSION_TAG="final_hallucination"
BENCHMARKS="chair,pope,amber,mme"
STUDENT_PX="${STUDENT_PX:-}"
TARGET_PX="${TARGET_PX:-448}"
DEGRADATION_MODE="${DEGRADATION_MODE:-}"
STUDENT_RATIO="${STUDENT_RATIO:-}"
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
        --student-px) STUDENT_PX="$2"; shift 2 ;;
        --target-px) TARGET_PX="$2"; shift 2 ;;
        --degradation-mode) DEGRADATION_MODE="$2"; shift 2 ;;
        --student-ratio) STUDENT_RATIO="$2"; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ${#OSS_NAMES[@]} -eq 0 ]]; then
    echo "Error: --oss-names is required" >&2
    exit 1
fi

case "${DEGRADATION_MODE:-square}" in
    square|original)
        ;;
    *)
        echo "Error: --degradation-mode must be square or original. Got: $DEGRADATION_MODE" >&2
        exit 1
        ;;
esac

normalize_benchmarks() {
    local raw="$1"
    local normalized=()
    local token
    raw="$(echo "$raw" | tr '[:upper:]' '[:lower:]' | tr -d ' ')"
    IFS=',' read -ra tokens <<< "$raw"
    for token in "${tokens[@]}"; do
        case "$token" in
            all)
                normalized+=(chair pope amber mme)
                ;;
            coco|frequent)
                normalized+=(chair pope)
                ;;
            final|external)
                normalized+=(amber mme)
                ;;
            chair|pope|amber|mme)
                normalized+=("$token")
                ;;
            "")
                ;;
            *)
                echo "Error: unsupported benchmark '$token'. Use chair,pope,coco,amber,mme,final,all." >&2
                exit 1
                ;;
        esac
    done

    local seen=","
    local unique=()
    for token in "${normalized[@]}"; do
        if [[ "$seen" != *",$token,"* ]]; then
            unique+=("$token")
            seen+="$token,"
        fi
    done
    local joined
    joined="$(IFS=','; echo "${unique[*]}")"
    echo "$joined"
}

BENCHMARKS="$(normalize_benchmarks "$BENCHMARKS")"
if [[ -z "$BENCHMARKS" ]]; then
    echo "Error: --benchmarks expanded to empty. Use chair,pope,coco,amber,mme,final,all." >&2
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

infer_eval_spec() {
    local exp_name="$1"
    local inferred_mode="square"
    local inferred_student_px="448"
    local inferred_student_ratio="1.0"

    if [[ "$exp_name" =~ -orig-sr([0-9.]+)-tr ]]; then
        inferred_mode="original"
        inferred_student_px="0"
        inferred_student_ratio="${BASH_REMATCH[1]}"
    elif [[ "$exp_name" =~ -s([0-9]+)(-|_) ]]; then
        inferred_mode="square"
        inferred_student_px="${BASH_REMATCH[1]}"
        inferred_student_ratio="1.0"
    fi

    echo "${DEGRADATION_MODE:-$inferred_mode}|${STUDENT_PX:-$inferred_student_px}|${TARGET_PX}|${STUDENT_RATIO:-$inferred_student_ratio}"
}

download_checkpoint() {
    local oss_path="$1"
    local local_ckpt_dir="$2"
    local f

    mkdir -p "$local_ckpt_dir" "${local_ckpt_dir}/actor/huggingface"
    ossutil cp "${oss_path}/model.safetensors" "${local_ckpt_dir}/model.safetensors" -f

    # vLLM reads --model from the checkpoint root, so HF metadata must live
    # beside model.safetensors. Keep actor/huggingface as a compatibility mirror.
    for f in "${HF_FILES[@]}"; do
        if ossutil cp "${oss_path}/${f}" "${local_ckpt_dir}/${f}" -f 2>/dev/null; then
            cp -f "${local_ckpt_dir}/${f}" "${local_ckpt_dir}/actor/huggingface/${f}"
        fi
    done

    if [[ ! -f "${local_ckpt_dir}/config.json" ]]; then
        echo "Error: config.json was not downloaded to ${local_ckpt_dir}." >&2
        echo "Check OSS path: ${oss_path}/config.json" >&2
        return 1
    fi
}

cleanup_checkpoint() {
    local local_ckpt_dir="$1"
    local f

    rm -f "${local_ckpt_dir}/model.safetensors" "${local_ckpt_dir}/model.safetensors.sha256"
    for f in "${HF_FILES[@]}"; do
        rm -f "${local_ckpt_dir}/${f}"
    done
    rm -rf "${local_ckpt_dir}/actor"
}

CURRENT_CKPT_DIR=""
cleanup_current_checkpoint() {
    if [[ -n "$CURRENT_CKPT_DIR" && -d "$CURRENT_CKPT_DIR" ]]; then
        echo "[cleanup] Removing current checkpoint files: $CURRENT_CKPT_DIR"
        cleanup_checkpoint "$CURRENT_CKPT_DIR"
    fi
}
trap cleanup_current_checkpoint EXIT

echo "=========================================="
echo " Final Hallucination Batch Eval from OSS"
echo " Step:        ${STEP}"
echo " Version tag: ${VERSION_TAG}"
echo " Benchmarks:  ${BENCHMARKS}"
echo " Student:     auto from experiment name unless overridden"
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
    IFS='|' read -r effective_degradation_mode effective_student_px effective_target_px effective_student_ratio < <(
        infer_eval_spec "$local_exp_name"
    )

    echo ""
    echo "=========================================="
    echo " Processing: ${oss_name}"
    echo " Local exp:  ${local_exp_name}"
    echo " OSS:        ${oss_path}"
    echo " Student:    mode=${effective_degradation_mode} px=${effective_student_px} target=${effective_target_px} ratio=${effective_student_ratio}"
    echo "=========================================="

    echo "[1/5] Downloading checkpoint ..."
    CURRENT_CKPT_DIR="$local_ckpt_dir"
    download_checkpoint "$oss_path" "$local_ckpt_dir"

    coco_tasks=()
    has_benchmark chair && coco_tasks+=(chair)
    has_benchmark pope && coco_tasks+=(pope)
    if [[ ${#coco_tasks[@]} -gt 0 ]]; then
        coco_mode="$(IFS=','; echo "${coco_tasks[*]}")"
        echo "[2/5] Running local COCO 300 (${coco_mode}) ..."
        DEGRADATION_MODE="$effective_degradation_mode" STUDENT_RATIO="$effective_student_ratio" TARGET_PX="$effective_target_px" \
            bash "$COCO_SCRIPT" "$local_ckpt_dir" "$effective_student_px" "$VERSION_TAG" "$coco_mode"
    fi

    if has_benchmark amber; then
        echo "[3/5] Running AMBER ..."
        STUDENT_PX="$effective_student_px" TARGET_PX="$effective_target_px" \
            DEGRADATION_MODE="$effective_degradation_mode" STUDENT_RATIO="$effective_student_ratio" \
            bash "$AMBER_SCRIPT" "$local_ckpt_dir" "$VERSION_TAG"
    fi

    if has_benchmark mme; then
        echo "[4/5] Running classic MME perception ..."
        STUDENT_PX="$effective_student_px" TARGET_PX="$effective_target_px" \
            DEGRADATION_MODE="$effective_degradation_mode" STUDENT_RATIO="$effective_student_ratio" \
            bash "$MME_SCRIPT" "$local_ckpt_dir" "$VERSION_TAG"
    fi

    echo "[5/5] Cleaning up model files ..."
    cleanup_checkpoint "$local_ckpt_dir"
    CURRENT_CKPT_DIR=""

    echo "Done: ${oss_name}"
done

trap - EXIT

echo ""
echo "=========================================="
echo " All final hallucination evals completed."
echo " Finished at: $(date)"
echo "=========================================="

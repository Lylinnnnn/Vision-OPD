#!/usr/bin/env bash

set -euo pipefail

step="${1:?Usage: $0 <all|50|100|150|156>}"
script_dir="$(cd "$(dirname "$0")" && pwd)"
case "$step" in
    all)
        for current_step in 50 100 150 156; do
            echo "===== InternVL3 tr0.5 ${INTERNVL_EVAL_EXPERIMENT:-low-res SFT} global_step_$current_step ====="
            bash "$script_dir/eval_internvl3_tr05_sft_step.sh" "$current_step"
        done
        exit 0
        ;;
    50|100|150|156) ;;
    *) echo "Unsupported checkpoint step: $step" >&2; exit 2 ;;
esac

res_opd_root="$(cd "$script_dir/.." && pwd)"
cd "$(dirname "$res_opd_root")"

export PYTHON_BIN="${PYTHON_BIN:-${CONDA_PREFIX:?Activate vision-opd first}/bin/python3}"
data_root="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"
export RES_OPD_DATA_ROOT="$data_root"
export BENCHMARK_DATA_DIR="${BENCHMARK_DATA_DIR:-$data_root}"
export VISION_BENCHMARK_DATA_DIR="$BENCHMARK_DATA_DIR"
export VISION_BENCHMARK=mmstar,cv-bench POPE_BENCHMARK=pope_adv,pope_pop,pope_random
export DATASET_VERSION=full RESULT_VERSION_TAG=internvl3 EVAL_OUTPUT_SUFFIX=official_v2
export CHAIR_PROMPT='Describe the image.'
export EVAL_SHARD_COUNT=8 GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"
export VLLM_BASE_PORT="${VLLM_BASE_PORT:-9300}"
export VLLM_PORT_CLEANUP=False VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1200}"
unset EVAL_OUTPUT_DIR VISION_BENCHMARK_OUTPUT_SUFFIX BENCHMARK_OUTPUT_SUFFIX

experiment="${INTERNVL_EVAL_EXPERIMENT:-Res-OPD-InternVL3-2B-hf-orig-sr1.0-tr0.5-sft-lowres-b32-full5k-e1}"
oss_name="${INTERNVL_EVAL_OSS_NAME:-ResOPD_InternVL3_2B_hf_orig_sr1.0_tr0.5_sft_lowres_b32_full5k-e1}"
source_dir="$data_root/ckpt/OPD/v4/$oss_name/global_step_$step"
oss_dir="${OSS_BASE:-oss://industry-algo/yanlin/ckpt/OPD/v4}/$oss_name/global_step_$step"
output_dir="$res_opd_root/eval_results/internvl3/full/${experiment}_global_step_${step}/train5000_test1000_original_sr1p0/official_v2"

results_complete() {
    [[ -s "$output_dir/chair_metrics.json" && -s "$output_dir/pope/pope_summary.json" && -s "$output_dir/amber/amber_metrics.json" ]] || return 1
    "$PYTHON_BIN" - "$output_dir" <<'PY' || return 1
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
chair = json.loads((root / "chair_metrics.json").read_text(encoding="utf-8"))
pope = json.loads((root / "pope/pope_summary.json").read_text(encoding="utf-8"))
amber = json.loads((root / "amber/amber_metrics.json").read_text(encoding="utf-8"))
complete = (
    chair.get("num_samples") == 1000
    and chair.get("total_results") == 1000
    and all(bench in pope.get("benchmarks", {}) for bench in ("pope_adv", "pope_pop", "pope_random"))
    and amber.get("official_returncode") == 0
)
sys.exit(0 if complete else 1)
PY
    "$PYTHON_BIN" "$res_opd_root/eval/check_vision_opd_eval.py" \
        --result-root "$output_dir" --benchmarks "$VISION_BENCHMARK" \
        --benchmark-data-dir "$BENCHMARK_DATA_DIR" >/dev/null || return 1
}

if [[ "${2:-}" == "--check-complete" ]]; then
    results_complete
    exit $?
fi

if results_complete; then
    echo "Complete evaluation already exists for global_step_$step: $output_dir"
    exit 0
fi

for required in model.safetensors model.safetensors.sha256 config.json tokenizer.json processor_config.json; do
    [[ -s "$source_dir/$required" ]] || {
        echo "Missing OSS checkpoint file: $source_dir/$required" >&2
        exit 1
    }
done
command -v ossutil >/dev/null || { echo "ossutil is required for local staging" >&2; exit 1; }
command -v flock >/dev/null || { echo "flock is required for step locking" >&2; exit 1; }
command -v sha256sum >/dev/null || { echo "sha256sum is required for checkpoint verification" >&2; exit 1; }

stage_root="${INTERNVL_STAGE_ROOT:-/tmp/resopd_internvl3_tr05_control_eval}"
stage_dir="$stage_root/$experiment/global_step_$step"
mkdir -p "$stage_dir"
exec 9>"$stage_root/${experiment}_step_$step.lock"
flock -n 9 || { echo "This step is already running on this machine" >&2; exit 1; }

for file in "$source_dir"/*; do
    [[ -f "$file" && "${file##*/}" != model.safetensors ]] || continue
    cp -f -- "$file" "$stage_dir/${file##*/}"
done
expected="$(awk 'NR == 1 {print $1}' "$stage_dir/model.safetensors.sha256")"
[[ "$expected" =~ ^[[:xdigit:]]{64}$ ]] || { echo "Invalid checkpoint SHA256 sidecar" >&2; exit 1; }
actual=""
if [[ -s "$stage_dir/model.safetensors" ]]; then
    actual="$(sha256sum "$stage_dir/model.safetensors" | awk '{print $1}')"
fi
if [[ "$actual" != "$expected" ]]; then
    available_kb="$(df -Pk "$stage_root" | awk 'NR == 2 {print $4}')"
    [[ "$available_kb" -ge 8388608 ]] || { echo "Less than 8 GiB free on /tmp" >&2; exit 1; }
    echo "Staging global_step_$step from OSS to node-local /tmp ..."
    ossutil cp "$oss_dir/model.safetensors" "$stage_dir/model.safetensors" -f
    actual="$(sha256sum "$stage_dir/model.safetensors" | awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || { echo "Staged model SHA256 mismatch" >&2; exit 1; }
fi

export EXPERIMENT_NAME_OVERRIDE="${experiment}_global_step_$step"
export DEGRADATION_MODE=original STUDENT_RATIO=1.0 TEACHER_RATIO=0.5
echo "Evaluating global_step_$step from node-local checkpoint: $stage_dir"
bash "$script_dir/eval_after_merge_sharded_8gpu.sh" "$stage_dir" internvl3 all

results_complete || { echo "Evaluation finished without complete metrics: $output_dir" >&2; exit 1; }
rm -rf -- "$stage_dir"
echo "Complete global_step_$step: $output_dir"

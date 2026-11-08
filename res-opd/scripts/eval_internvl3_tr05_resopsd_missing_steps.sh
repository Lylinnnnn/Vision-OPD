#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$script_dir/../.." && pwd)"
python="${PYTHON_BIN:-${CONDA_PREFIX:?Activate vision-opd first}/bin/python3}"
data_root="${RES_OPD_DATA_ROOT:-/data/oss_bucket_0/yanlin}"
experiment=Res-OPD-InternVL3-2B-hf-tr05-fixedmb1-full5k-e1
oss_name=ResOPD_InternVL3_2B_hf_tr05_fixedmb1_full5k-e1
gpu_list="${GPU_LIST:-0,1,2,3,4,5,6,7}"
base_port="${VLLM_BASE_PORT:-23100}"
lock="$root/res-opd/eval_results/internvl3/.${experiment}.missing_steps.lock"

[[ -x "$python" ]] || { echo "Python not executable: $python" >&2; exit 1; }
mkdir -p "$(dirname "$lock")"
command -v flock >/dev/null || { echo "flock is required" >&2; exit 1; }
exec 9>"$lock"
flock -n 9 || { echo "This InternVL3 Res-OPSD fill is already running" >&2; exit 1; }

export PYTHON_BIN="$python" RES_OPD_DATA_ROOT="$data_root"
export INTERNVL_EVAL_EXPERIMENT="$experiment" INTERNVL_EVAL_OSS_NAME="$oss_name"
export INTERNVL_STAGE_ROOT="${INTERNVL_STAGE_ROOT:-/tmp/resopd_internvl3_resopsd_fill}"
export GPU_LIST="$gpu_list" VLLM_BASE_PORT="$base_port"
export VLLM_PORT_CLEANUP=False VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.28}"
export VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-1800}"

for step in 50 100 150; do
    source_dir="$data_root/ckpt/OPD/v4/$oss_name/global_step_$step"
    for file in model.safetensors model.safetensors.sha256 config.json tokenizer.json processor_config.json; do
        [[ -s "$source_dir/$file" ]] || { echo "Missing checkpoint file: $source_dir/$file" >&2; exit 1; }
    done
done
echo "INTERNVL3_RESOPSD_CHECKPOINTS=PASS"

for step in 50 100 150; do
    echo "===== InternVL3 Res-OPSD global_step_$step ====="
    if bash "$script_dir/eval_internvl3_tr05_sft_step.sh" "$step" --check-complete; then
        echo "SKIP: global_step_$step already complete"
        continue
    fi
    "$python" "$script_dir/tmp/wait_eval_capacity.py" \
        --gpu-list "$gpu_list" --base-port "$base_port" --port-count 8 \
        --min-free-mib "${MIN_FREE_MIB:-36000}" --wait-seconds "${CAPACITY_WAIT_SECONDS:-7200}"
    bash "$script_dir/eval_internvl3_tr05_sft_step.sh" "$step"
done
echo "INTERNVL3_RESOPSD_STEPS=PASS"

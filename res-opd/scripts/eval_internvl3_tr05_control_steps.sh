#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 {lowres_sft|frozen_rkl|randommask|entropymask} {all|50|100|150|156}" >&2
    exit 2
fi

method="$1"
step="$2"
case "$method" in
    lowres_sft)
        experiment=Res-OPD-InternVL3-2B-hf-orig-sr1.0-tr0.5-sft-lowres-b32-full5k-e1
        oss_name=ResOPD_InternVL3_2B_hf_orig_sr1.0_tr0.5_sft_lowres_b32_full5k-e1
        ;;
    frozen_rkl)
        experiment=Res-OPD-InternVL3-2B-hf-orig-sr1.0-tr0.5-a1.0-frozen-rkl-b32-rn4-full5k-e1
        oss_name=ResOPD_InternVL3_2B_hf_orig_sr1.0_tr0.5_a1.0_frozen_rkl_b32_rn4_full5k-e1
        ;;
    randommask)
        experiment=Res-OPD-InternVL3-2B-hf-orig-sr1.0-tr0.5-a1.0-frozen-rkl-randommask-p30-b32-rn4-full5k-e1
        oss_name=ResOPD_InternVL3_2B_hf_orig_sr1.0_tr0.5_a1.0_frozen_rkl_randommask_p30_b32_rn4_full5k-e1
        ;;
    entropymask)
        experiment=Res-OPD-InternVL3-2B-hf-orig-sr1.0-tr0.5-a1.0-frozen-rkl-entropymask-p30-b32-rn4-full5k-e1
        oss_name=ResOPD_InternVL3_2B_hf_orig_sr1.0_tr0.5_a1.0_frozen_rkl_entropymask_p30_b32_rn4_full5k-e1
        ;;
    *)
        echo "Unsupported method: $method" >&2
        exit 2
        ;;
esac

case "$step" in
    all|50|100|150|156) ;;
    *) echo "Unsupported checkpoint step: $step" >&2; exit 2 ;;
esac

script_dir="$(cd "$(dirname "$0")" && pwd)"
export INTERNVL_EVAL_EXPERIMENT="$experiment" INTERNVL_EVAL_OSS_NAME="$oss_name"
exec bash "$script_dir/eval_internvl3_tr05_sft_step.sh" "$step"

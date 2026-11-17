# RLSF-V on the Res-OPD COCO 5k Split

The unmodified RLSF-V source is vendored at `res-opd/third_party/baselines/RLSF-V/`.
It comes from https://github.com/XLearning-SCU/RLSF-V (MIT license); the source
ZIP supplied on 2026-11-17 has SHA-256
`d0a61a41e77bf2ae337052d36e8f915d83e37c826009dd8631194a07b2f6b7bb`. The nested
LlamaFactory source retains its own license. The adapter scripts in this folder
are separate from that upstream source.

## Comparison Protocol

- Start each run from the corresponding Base model, not a Res-OPD checkpoint.
- Use the exact 5,000 training images and prompt from `res-opd/data/train_5k.parquet`.
  `prepare_coco5k.py` remaps image paths to the mounted COCO train2017 directory
  and rejects overlap with `test_1000.json`.
- Keep RLSF-V's 10 sampled answers per prompt, top-50 saved logits, fuzzy
  semantic feedback, top-R=5 for Qwen and top-R=40 for InternVL, and LoRA-DPO
  settings. For reliability, the generated training config saves each epoch.
- Evaluate exported adapters using the existing 1,000-image CHAIR, three POPE
  splits, and AMBER protocol. Results are separate from the Base/Res-OPD runs.
- Do not present the released Qwen3-VL-8B RLSF-V weight as a matched-data run:
  it was trained with the upstream RLAIF-V 10k question set.

## Environment

Use separate environments. Do not replace packages in `vision-opd`.
Follow `third_party/baselines/RLSF-V/train/README.md` for the upstream
generation/conversion environment (`transformers==4.57.3`, spaCy
`en_core_web_sm`) and the upstream LlamaFactory environment. Install the
vendored LlamaFactory from its own directory in that second environment.
`prepare` needs pandas/pyarrow; `train` and `export` need PyYAML and the
`llamafactory-cli` executable. `eval` uses the existing `vision-opd` environment.
The `preflight` phase checks all generation/conversion imports and the spaCy
model before starting GPU workers. It reports every missing dependency found.
If `conda activate rlsf-v-train` fails, stop there; do not silently continue in
`vision-opd`.

## Per-Model Phases

From the Vision-OPD repository root, set `PYTHON_BIN` to the active environment's
Python, then run each phase separately:

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b prepare
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b preflight
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b generate
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b convert
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b train
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b export
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b infer-smoke
PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b eval
```

Replace `qwen2b` with `qwen4b`, `qwen8b`, or `internvl2b`. The shared
`prepare` phase only needs to be run once. The default paths target the
server's model tree and `/data/oss_bucket_0/yanlin`; override `MODEL_PATH`,
`RES_OPD_DATA_ROOT`, or `RLSF_V_WORK_ROOT` if necessary. Generated artifacts
are stored under `res-opd/checkpoints/rlsf_v_coco5k/` by default.
The `train` phase first writes locally, then automatically uploads and checks
every adapter/checkpoint file under
`oss://industry-algo/yanlin/ckpt/OPD/v4/RLSF_V_COCO5K/rlsf_v_coco5k/<model>/adapter/`.
It leaves local files intact. Upload failure makes the phase fail; `sync-adapter`
retries the upload without retraining. Set `RLSF_V_SYNC_OSS=0` for a local-only
smoke run. Exported full merged weights remain local by default because they
can be regenerated from Base + adapter; set `RLSF_V_SYNC_MERGED_OSS=1` before
`export` to upload them as well. Each uploaded file is verified by OSS size.

Before a full run, use a separate work root for a two-image smoke test:

```bash
RLSF_V_WORK_ROOT="$PWD/res-opd/checkpoints/rlsf_v_smoke" RLSF_V_MAX_QUESTIONS=2 \
  PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b prepare
RLSF_V_WORK_ROOT="$PWD/res-opd/checkpoints/rlsf_v_smoke" \
  PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b generate
```

Run `convert`, `train` (in the separate LlamaFactory environment with
`RLSF_V_SYNC_OSS=0`), `export`, then `infer-smoke` (in the generation
environment) in the smoke work root before attempting a full run.
`infer-smoke` generates one short answer for each of two COCO train images;
it is only a model-load/response check, not a benchmark score.
The upstream Qwen3-VL generator was released for 8B and the InternVL
generator for InternVL3.5-8B. Their 2B/4B and InternVL3-2B use here are ports,
not upstream-validated reproductions; GPU smoke is required before full runs.

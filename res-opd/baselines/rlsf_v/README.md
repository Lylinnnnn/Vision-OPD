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
If `conda activate rlsf-v-lf` fails, stop there; do not silently continue in
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
The `train` phase saves once per epoch (two epochs), keeps at most one
intermediate checkpoint, and omits optimizer state from saved checkpoints.
If this model's DPO file is absent, `train` first runs `generate` and `convert`
with the `vision-opd` Python at
`/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3` (override with
`RLSF_V_GENERATION_PYTHON_BIN`). A full work root requires exactly 5000
questions. Existing incomplete generation outputs are never overwritten;
inspect or resume them before restarting training.
It uploads all adapter files under
`oss://industry-algo/yanlin/ckpt/OPD/v4/RLSF_V_COCO5K/rlsf_v_coco5k/<model>/adapter/`.
After every file and the completion manifest are verified on OSS, it deletes
only this model's local `adapter/` tree, including the final adapter. If upload
fails, it deletes nothing; `sync-adapter` retries without retraining. `export`
restores the final adapter from OSS (without epoch checkpoints), merges it with
Base, then uploads the merged model and deletes its local copy after verification.
`infer-smoke` and `eval` each restore the merged model to a model-specific
temporary directory and remove that directory on exit. Set
`RLSF_V_KEEP_LOCAL_AFTER_SYNC=1` to retain local uploaded artifacts, or
`RLSF_V_SYNC_MERGED_OSS=0` to keep the merged model local after export; the
latter disables OSS-backed inference until `sync-merged` succeeds. Generated
preferences, logs, and training configuration are not deleted. The automatic
cleanup is restricted to the four model names under `rlsf_v_coco5k`,
`rlsf_v_coco5k_e1`, or `rlsf_v_smoke`; it never scans other checkpoint directories.

Before a full run, use a separate work root for a two-image smoke test:

```bash
RLSF_V_WORK_ROOT="$PWD/res-opd/checkpoints/rlsf_v_smoke" RLSF_V_MAX_QUESTIONS=2 \
  PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b prepare
RLSF_V_WORK_ROOT="$PWD/res-opd/checkpoints/rlsf_v_smoke" \
  PYTHON_BIN="$CONDA_PREFIX/bin/python3" bash res-opd/baselines/rlsf_v/run_coco5k.sh qwen2b generate
```

Run `convert`, `train` (in the separate LlamaFactory environment), `export`,
then `infer-smoke` (in the generation
environment) in the smoke work root before attempting a full run.
`infer-smoke` generates one short answer for each of two COCO train images;
it is only a model-load/response check, not a benchmark score.
After training, run `export` in `rlsf-v-lf` to restore the adapter from OSS,
merge it with Base, verify its upload, and remove the local merged model. Run
`eval` in `vision-opd` with `RLSF_V_EVAL_MODE=chair` to restore the merged model
temporarily and reuse `eval_after_merge_sharded_8gpu.sh` for 1000-image CHAIR
inference and official-synonym metrics. Its output is under
`eval_results/external_baselines/full/<base-model>/train5000_test1000_original_sr1p0/rlsf_v_coco5k_official_v2/`.
After CHAIR, `RLSF_V_EVAL_MODE=pope,amber` runs just those two benchmarks and
adds their official metrics to the same directory without redoing CHAIR.
For a matched one-epoch control, set `RLSF_V_WORK_ROOT` to the sibling
`rlsf_v_coco5k_e1` directory and run `train`, `export`, then `eval` with
`RLSF_V_EVAL_MODE=chair,pope,amber`. The control reads the original 5k questions
and per-model DPO file without regenerating preferences, fixes training to one
epoch, uses a separate OSS prefix, and writes metrics under
`rlsf_v_coco5k_e1_official_v2`. The two-epoch adapter is never modified.
To reevaluate an existing adapter without overwriting prior metrics, set
`RLSF_V_EVAL_SUFFIX` to a unique alphanumeric label. The label is appended
to the result directory; an existing labeled directory is never reused.
The upstream Qwen3-VL generator was released for 8B and the InternVL
generator for InternVL3.5-8B. Their 2B/4B and InternVL3-2B use here are ports,
not upstream-validated reproductions; GPU smoke is required before full runs.

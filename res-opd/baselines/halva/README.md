# HALVA Adaptation for Res-OPD

This is an **adaptation**, not an unmodified run of the original LLaVA/VILA
implementation. HALVA's official repository is
[pritamqu/HALVA](https://github.com/pritamqu/HALVA), commit
`059b086f7b4d3d5af436f043d5c9eb9cc627c3e0`. The paper is
[arXiv:2405.18654](https://arxiv.org/abs/2405.18654). Its LLaVA training
entry points cannot load Qwen3VL or HF InternVL3 models directly.
Run `bash res-opd/baselines/halva/fetch_upstream.sh` to place the pinned
official source alongside other external baselines. It is reference code;
the adaptation does not import its LLaVA-specific modules.

The adaptation retains the official loss shape: `softplus(negative phrase
log-probability - positive phrase log-probability) + 0.4 * KL(base || adapted)`.
The Base is the same frozen model with LoRA disabled. The original uses Visual
Genome augmented samples and a separate LLaVA reference subset; here we use
the matched Res-OPD COCO 5k training images, their COCO train2017 reference
captions and instance categories, and the same training prompt. We replace
one annotated object phrase with an absent object from the same COCO
supercategory when available (otherwise another unannotated COCO class). If
no reference caption contains a unique annotated class name, preparation
appends a short class-name clause to a reference caption. When every annotated
class name already occurs multiple times, it uses only the class-name clause
so phrase alignment remains unambiguous. Each pair records its `strategy`, and
`prepare` prints the strategy counts. Images without an annotated object or a
reference caption cannot yield a pair; the script never silently trains on
fewer than the requested 5k images. This is a reproducible substitution
baseline, not a claim that the original generative augmentation distribution
was reproduced exactly.

For the current matched COCO 5k list, 38 images have no instance annotations.
To deliberately use the other 4,962 images, set `HALVA_EXPECTED_QUESTIONS=5000`
and `HALVA_EXPECTED_PAIRS=4962` for `prepare`. The preparation only accepts
this reduced count when every skipped image lacks instance annotations; any
other mismatch still fails. Report the pair count and strategy breakdown with
results, rather than describing the run as 5k training pairs.

Required files on the server:

- `res-opd/checkpoints/rlsf_v_coco5k/questions.jsonl` (the matched 5k images
  and prompts; create with the existing RLSF-V `prepare` phase if absent).
- `$RES_OPD_DATA_ROOT/COCO/annotations/captions_train2017.json` and
  `instances_train2017.json`, plus the COCO training images referenced by
  the questions file. Override annotation paths with `HALVA_CAPTIONS` and
  `HALVA_INSTANCES` if the OSS layout differs.
- Local Qwen3VL 2B/4B/8B or HF InternVL3-2B Base model, selected by argument
  or `HALVA_MODEL_PATH`.

Use `rlsf-v-lf` Python for training/export (`torch`, `transformers`, `peft`,
`Pillow`) and `vision-opd` Python for evaluation (`vllm` and benchmark
dependencies). `HALVA_TRAIN_PYTHON_BIN` and `HALVA_EVAL_PYTHON_BIN` override
the defaults. The smoke uses one GPU and two examples, trains one step, exports
merged weights, and invokes the existing sharded CHAIR/POPE/AMBER evaluation.
The two-row AMBER smoke validates inference and merge only; official AMBER
scoring requires a full benchmark run.

```bash
cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD
HALVA_RUN_NAME=halva_smoke_qwen2b_$(date +%Y%m%d_%H%M%S) \
  HALVA_GPU_LIST=0 HALVA_BASE_PORT=35400 \
  bash res-opd/baselines/halva/run_coco5k.sh qwen2b smoke
```

For a full run, invoke `prepare`, `train`, `export`, and `eval` in that order
with the same `HALVA_RUN_NAME`. The eval phase uses the existing 8-shard
pipeline and writes under
`res-opd/eval_results/external_baselines/full/<model>/train5000_test1000_original_sr1p0/<run_name>_official_v2`.
Current adapter and merged checkpoint artifacts stay in the run-specific local
directory. They are **not** automatically uploaded or deleted; do not start
large full runs until OSS retention is arranged.

## Resume an interrupted evaluation

`resume_eval.py` reuses the original eight shard answer directories and writes
the merged metrics to the same formal `eval_results` directory. It locates one
shard run under the formal `.shards` directory or an already staged
`/tmp/halva_*_resume_*` directory. If several are found, pass the exact path
with `--shard-root`; it never combines runs. Shared-home shard data is moved
to `/tmp` before resuming to avoid filling the home quota. The script repairs
an incomplete final JSONL line, reconstructs AMBER's response index from
valid raw answers, and archives old exit codes and logs. CHAIR and POPE use
their existing resume logic. It does not export a model again.

First inspect what will be used:

```bash
python3 res-opd/baselines/halva/resume_eval.py qwen2b --dry-run
```

On a machine dedicated to this evaluation, use the following command. The
explicit cleanup flag stops **all GPU processes owned by the current user on
that machine** and the old sharded parent for this merged model. Occupied ports
belonging to unrelated services are left alone. Do not use this flag while
another training or evaluation job is running.

```bash
python3 res-opd/baselines/halva/resume_eval.py qwen2b --clear-user-gpus
```

Replace `qwen2b` with `qwen4b`, `qwen8b`, or `internvl2b` on the other machines.
Use `--run-name`, `--base-port`, `--gpu-list`, or `--shard-root` when the original
run used non-default values. The merged model must still be available under
`res-opd/checkpoints/halva_coco5k/<run>/<model>/merged`. The script keeps the
staged `/tmp` shard answers after a successful run for auditing. A `PASS`
requires all expected CHAIR, POPE, and AMBER rows and successful official
AMBER scoring; error-derived answers require a separate quality audit.
The script searches subsequent 100-port blocks if any of its eight startup
ports are occupied. If a port conflict occurs after startup, it preserves the
shards and retries with another free block (up to three attempts by default,
configurable with `--port-attempts`). Non-port failures are not retried.

## Inspect main-table metrics

The general read-only viewer under `res-opd/eval/` accepts any combination of
evaluation directories, not just HALVA runs. Pass an optional label per row:

```bash
/home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 res-opd/eval/show_eval_metrics.py \
  "HALVA-4B=res-opd/eval_results/external_baselines/full/Qwen3VL-4B-Instruct/train5000_test1000_original_sr1p0/halva_coco5k_e1_official_v2" \
  "HALVA-8B=res-opd/eval_results/external_baselines/full/Qwen3VL-8B-Instruct/train5000_test1000_original_sr1p0/halva_coco5k_e1_official_v2"
```

Missing metrics are shown as `--`. The audit lines show answer counts, metric
schema, and error-derived answers; such scores remain provisional until reviewed.

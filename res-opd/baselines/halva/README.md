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
supercategory. This is a reproducible substitution baseline, not a claim that
the original generative augmentation distribution was reproduced exactly.

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

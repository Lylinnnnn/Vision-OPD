# APASI-Base on the Res-OPD COCO 5k Split

This is a **model-family and data adaptation**, not the released APASI-Base-7B
checkpoint or a claim to reproduce its published numbers. The
[paper](https://aclanthology.org/2025.findings-emnlp.746/) and
[official code](https://github.com/davidluciolu/APASI) use LLaVA-v1.5-7B and SI-23k. This
adaptation starts independently from each Qwen3VL 2B/4B/8B or HF InternVL3-2B
Base model, uses the same disjoint COCO 5k training images and prompt as the
other Res-OPD external baselines, and evaluates on the existing CHAIR/POPE/
AMBER protocol. Do not mix these metrics with the original paper's 500-image
Object-Hal numbers.

The official source is fetched, not imported as LLaVA model code. The pinned
commit is `849fa4c212e6fc97c7b56abdfe65d20b1d73292b`. Its LVIS vocabulary
and `co_graph_construction.py` parse model-generated preferred captions with
WordNet/NLTK to construct the object co-occurrence graph. The adaptation then
selects later sentences with the upstream square-root weighting, chooses an
absent object from the graph's top-ten neighbors, and lets the **same target
model without the image** complete a guided hallucinated sentence. The
preferred caption and injected response form one DPO pair. This is the
single-iteration APASI-Base setting (no curriculum); some images may be
ineligible because their captions are too short or contain no usable object.
The actual pair count is printed and stored in `dataset/preferences.json`.

The DPO config uses beta 0.1, learning rate `4e-7`, LoRA rank/alpha 128/256,
and three epochs to follow the upstream APASI-Base script. The effective
8-GPU batch is 64 (`per_device_train_batch_size=1`, accumulation 8). The
`cutoff_len` is 4096, increased from upstream's 2048 because the adapted
image processors can expand a single image to over 3000 tokens. These model
and data changes must be disclosed alongside any comparison. Override
`APASI_EPOCHS` and `APASI_CUTOFF_LEN` only with a separately named run.

## Environment

- Generation/evaluation: existing `vision-opd` Python with PyTorch,
  Transformers, Pillow, pandas, pyarrow, NLTK, and vLLM.
- Training/export: existing `rlsf-v-lf` environment with the vendored
  LlamaFactory. Do not install it into `vision-opd`.
- NLTK resources: `punkt_tab`, `averaged_perceptron_tagger_eng`, and
  `wordnet`. `preflight` checks them before GPU loading. Download into a
  user-controlled `NLTK_DATA` directory if the server lacks them.
- `ossutil` for verified upload/restore. The APASI storage wrapper accepts
  only `apasi_coco5k` or `apasi_smoke` work roots and these four model names.

The default COCO image root is `/data/oss_bucket_0/yanlin`; set
`RES_OPD_DATA_ROOT` to a different mounted root if needed. `prepare` checks
every training image and rejects overlap with `test_1000.json`. The graph
builder fetches the pinned APASI source under
`res-opd/third_party/baselines/APASI/`; it does not install LLaVA.

## End-to-end smoke

From the repository root in the `rlsf-v-lf` shell, use a unique smoke run
name. The smoke uses 64 training images and one GPU for generation and a
one-step DPO check, then evaluates two CHAIR/POPE/AMBER samples with official
AMBER scoring intentionally skipped. It is a pipeline check, not a metric.

```bash
cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD
conda activate rlsf-v-lf
export RES_OPD_DATA_ROOT=/data/oss_bucket_0/yanlin
export APASI_RUN_NAME="apasi_smoke_qwen2b_$(date +%Y%m%d_%H%M%S)"
export APASI_GPU_LIST=0
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b smoke
```

For the full 5k run, use a new `APASI_RUN_NAME` and run phases separately:

```bash
export APASI_RUN_NAME=apasi_base_coco5k_qwen2b
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b prepare
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b preflight
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b generate
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b train
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b export
bash res-opd/baselines/apasi/run_coco5k.sh qwen2b eval
```

Set `APASI_GPU_LIST=0,1,2,3,4,5,6,7` for the full eight-shard run and
replace `qwen2b` with `qwen4b`, `qwen8b`, or `internvl2b` on separate machines.
`generate` retains completed JSONL rows by shard and skips them after an
interruption; do not change the shard count/list when resuming. Each phase
has logs under its own run directory. Training and export upload verified
artifacts to `oss://industry-algo/yanlin/ckpt/OPD/v4/APASI_COCO5K/<run>/<model>/`
and then delete only that artifact's local directory. Evaluation restores the
merged weights under `/tmp/apasi_coco5k`, picks an unused port block, invokes
the existing eight-shard CHAIR/POPE/AMBER script, and cleans the staged model
only after success. A failed evaluation retains the staged model and shard
answers for diagnosis; no other GPU processes are killed.

Formal results are under
`res-opd/eval_results/external_baselines/full/<base-model>/train5000_test1000_original_sr1p0/<run>_official_v2`.
The separate `res-opd/eval/show_eval_metrics.py` can display them alongside
Base and other methods without rewriting any benchmark scorer.

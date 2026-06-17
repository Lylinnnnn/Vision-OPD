# Res-OPD Validation Benchmarks

Use `res-opd/scripts/eval_after_merge.sh` after a checkpoint has been merged.
The script starts vLLM, runs the selected benchmark group, writes raw outputs,
and collects a compact summary.

## Benchmark Groups

- `chair`: COCO caption hallucination metrics on `res-opd/data/test.json`.
- `pope`: POPE-style yes/no object probes built from the same Res-OPD COCO test split by default.
- `frequent`: `chair,pope`.
- `vision`: Vision-OPD's reusable API eval stack, controlled by `VISION_BENCHMARK`.
- `all`: `chair,pope,vision`.

Recommended default while training short runs:

```bash
bash res-opd/scripts/eval_after_merge.sh <merged_checkpoint_path> 0 latest frequent
```

Recommended final or key-checkpoint run:

```bash
VISION_BENCHMARK="mmstar,pope_adv,pope_pop,pope_random" \
JUDGE_API_BASE="http://<judge-server>/v1/" \
JUDGE_MODEL="<judge-model>" \
bash res-opd/scripts/eval_after_merge.sh <merged_checkpoint_path> 0 latest all
```

`VISION_BENCHMARK` can use the benchmarks supported by the upstream
Vision-OPD eval code, including `vstar`, `zoombench`, `hrbench-4k`,
`hrbench-8k`, `mme-realworld`, `mme-realworld-cn`, `mme-realworld-lite`,
`visualprobe`, `mmvp`, `cv-bench`, `mmstar`, and POPE splits.

## Outputs

For each checkpoint, outputs are stored under:

```text
res-opd/eval_results/<version_tag>/<experiment_name>_<step_tag>/<dataset_tag>/
```

Important files:

- `chair_metrics.json`
- `pope/pope_summary.json`
- `vision_opd/model_answer/...`
- `vision_opd/judge/...`
- `compact_summary.json`
- `compact_summary.csv`

`compact_summary` is regenerated at the end of every run. If one benchmark
fails, completed outputs from earlier benchmarks are kept and summarized.

## Final Hallucination Benchmarks

Run these after training on selected merged checkpoints.

### AMBER

AMBER uses official-format responses:

- generative: `{"id": <id>, "response": "<caption>"}`
- discriminative: `{"id": <id>, "response": "Yes"|"No"}`

The wrapper generates responses and then calls the official AMBER
`inference.py` unless `AMBER_SKIP_OFFICIAL_EVAL=True` is set. Official AMBER
requires `spacy`, `nltk`, and the `en_core_web_lg` model.

```bash
AMBER_ROOT=/path/to/AMBER \
AMBER_IMAGE_ROOT=/path/to/AMBER/images \
bash res-opd/scripts/tmp/val_amber.sh <merged_checkpoint_path> latest
```

For low-disk machines, stage AMBER from OSS into
`/home/liuyanlin.lyl/notebook/data/AMBER` only for the current run:

```bash
AMBER_OSS_URI=oss://<bucket>/<path>/AMBER \
bash res-opd/scripts/tmp/val_amber.sh <merged_checkpoint_path> latest
```

Useful knobs:

- `AMBER_EVAL_TYPE=a`: all AMBER tasks; use `g`, `d`, `de`, `da`, or `dr` for subsets.
- `AMBER_SKIP_OFFICIAL_EVAL=True`: only generate the official response JSON.
- `AMBER_MAX_SAMPLES=100`: quick smoke test.
- `KEEP_BENCHMARK_DATA=True`: keep staged data after eval; default removes data staged by the script.

### Classic MME Perception

This is classic MME, not MME-RealWorld. The default categories are:

```text
existence,count,position,color
```

The evaluator follows the common MME scoring style: each category reports
`accuracy`, `accuracy_plus`, and `score = accuracy + accuracy_plus`, with
perception/cognition/total score aggregation.

```bash
MME_ROOT=/path/to/MME_Benchmark_release_version \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

For low-disk machines, stage classic MME from OSS into
`/home/liuyanlin.lyl/notebook/data/MME_Benchmark_release_version` only for
the current run:

```bash
MME_OSS_URI=oss://<bucket>/<path>/MME_Benchmark_release_version \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

If you already have converted JSON/JSONL with `image`, `question`, `answer`,
and `category` fields:

```bash
MME_JSON=/path/to/mme_perception.json \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

Outputs for both AMBER and MME are placed under:

```text
res-opd/eval_results/<version>/<experiment>_<step>/final_hallucination/
```

### OSS Dataset Cache

The temporary scripts can use one shared OSS base:

```bash
BENCHMARK_OSS_BASE=oss://<bucket>/<path>/benchmarks \
bash res-opd/scripts/tmp/eval_hallucination_batch_from_oss.sh \
  --oss-names <exp> --step global_step_40 --benchmarks amber,mme
```

Expected OSS layout:

```text
oss://<bucket>/<path>/benchmarks/AMBER/
oss://<bucket>/<path>/benchmarks/MME_Benchmark_release_version/
```

After a manual first download, upload with:

```bash
ossutil cp -r /home/liuyanlin.lyl/notebook/data/AMBER/ \
  oss://<bucket>/<path>/benchmarks/AMBER/ -f
ossutil cp -r /home/liuyanlin.lyl/notebook/data/MME_Benchmark_release_version/ \
  oss://<bucket>/<path>/benchmarks/MME_Benchmark_release_version/ -f
```

ModelScope fallback is configurable but intentionally not hard-coded:

```bash
AMBER_MODELSCOPE_ID=<owner/dataset>
MME_MODELSCOPE_ID=<owner/dataset>
```

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

## OPD Student/Teacher Distribution Trace

There are two related but different trace workflows.

### Post-Merge Test/Eval Trace

Use this when a checkpoint has already been merged and `eval_chair.py` has
produced `eval_results.jsonl` on the test split. The scorer does not run vLLM
generation again. It reads fixed captions and runs forced HF forwards with the
student image view and the teacher image view, then writes token-level
student/teacher logprob, top-k, and optional entropy records.

One-command path through `eval_after_merge.sh`:

```bash
EVAL_MODE=chair \
EVAL_OPD_TRACE=True \
EVAL_OPD_TRACE_TOPK=100 \
EVAL_OPD_TRACE_ENTROPY=True \
EVAL_OPD_TRACE_CASE_ANALYSIS=res-opd/eval_results/v2/case_analysis/sr1.0-tr0.25_vs_baseline/all_cases_sorted.json \
bash res-opd/scripts/eval_after_merge.sh <merged_checkpoint_path> 0 v2 chair
```

If `eval_results.jsonl` already exists and you only want the student/teacher
distribution trace, run the scorer directly:

```bash
python res-opd/eval/score_opd_eval_trace.py \
  --model-path <merged_checkpoint_path> \
  --eval-results <eval_results.jsonl> \
  --output-jsonl <output_dir>/opd_eval_trace.jsonl \
  --case-analysis res-opd/eval_results/v2/case_analysis/sr1.0-tr0.25_vs_baseline/all_cases_sorted.json \
  --degradation-mode original \
  --student-ratio 1.0 \
  --teacher-ratio 0.25 \
  --topk 100 \
  --entropy \
  --trace-scope eval

python res-opd/eval/analyze_opd_trace.py \
  --trace-dir <output_dir>/opd_eval_trace.jsonl \
  --case-analysis res-opd/eval_results/v2/case_analysis/sr1.0-tr0.25_vs_baseline/all_cases_sorted.json \
  --output-json <output_dir>/opd_eval_trace_summary.json \
  --output-md <output_dir>/opd_eval_trace_summary.md
```

For a quick smoke test, use `--topk 20 --max-samples 100` first. For the three
original-ratio teacher settings, change both `--teacher-ratio` and the
case-analysis directory:

```text
tr0.25 -> --teacher-ratio 0.25 -> case_analysis/sr1.0-tr0.25_vs_baseline/
tr0.5  -> --teacher-ratio 0.5  -> case_analysis/sr1.0-tr0.5_vs_baseline/
tr0.75 -> --teacher-ratio 0.75 -> case_analysis/sr1.0-tr0.75_vs_baseline/
```

The most important summary fields for selective hallucination suppression are:

- `object_trace_summary.mention_type_summary.correct_object`
- `object_trace_summary.mention_type_summary.hallucinated_object`
- `object_trace_summary.correct_vs_hallucinated_signal`
- `object_trace_summary.step_contrast_hallucinated_minus_correct`

Positive evidence means hallucinated-object mentions have a more negative
`teacher_minus_student_selected_logprob_mean`, or a higher
`teacher_selected_logprob_lt_student_frac`, than correct-object mentions.

### Training Mini-Eval Generations

Use this when you want step-by-step changes during training. The training loop
uses the fixed `res-opd/data/val.parquet` split, not the current train batch.
At every validation step it writes raw mini-eval generations to:

```text
res-opd/mini_eval_generations/<experiment_name>/<global_step>.jsonl
```

These files are light text dumps with captions and metadata. They are not the
large token-level forced-scoring traces yet. After a checkpoint is merged, feed
the corresponding mini-eval generation file to `score_opd_eval_trace.py` with
`--trace-scope mini_eval`.

Recommended training knobs for a 2-epoch run:

```bash
TOTAL_EPOCHS=2 \
OPD_MINI_EVAL_TRACE=True \
OPD_MINI_EVAL_TEST_FREQ=10 \
OPD_MINI_EVAL_MAX_SAMPLES=50 \
VAL_N=3 \
bash res-opd/scripts/run_res_opd.sh
```

For a 1-epoch sanity run, use `OPD_MINI_EVAL_TEST_FREQ=5` to get enough points.
For 2 epochs, `OPD_MINI_EVAL_TEST_FREQ=10` is usually enough. The raw generation
files remain small; the expensive part is the later forced scoring.

Example forced scoring for one mini-eval step:

```bash
python res-opd/eval/score_opd_eval_trace.py \
  --model-path <merged_checkpoint_path>/global_step_50 \
  --eval-results res-opd/mini_eval_generations/<experiment_name>/50.jsonl \
  --output-jsonl res-opd/eval_results/<version>/<experiment_name>_mini_eval/opd_trace_steps.jsonl \
  --checkpoint-step 50 \
  --trace-scope mini_eval \
  --degradation-mode original \
  --student-ratio 1.0 \
  --teacher-ratio 0.5 \
  --topk 100 \
  --entropy
```

Append several scored steps into the same JSONL, then analyze the combined file:

```bash
python res-opd/eval/analyze_opd_trace.py \
  --trace-dir res-opd/eval_results/<version>/<experiment_name>_mini_eval/opd_trace_steps.jsonl \
  --output-json res-opd/eval_results/<version>/<experiment_name>_mini_eval/opd_trace_summary.json \
  --output-md res-opd/eval_results/<version>/<experiment_name>_mini_eval/opd_trace_summary.md
```

`analyze_opd_trace.py` will report `global_steps`, `trace_coverage.step_summary`,
and object-level step trends when GT objects are present in the trace metadata.

### OSS Notes

When `POST_TRAIN_SYNC_TO_OSS=True`, `run_res_opd.sh` uploads mini-eval
generation dumps with the other training artifacts:

```text
<oss_exp_path>/training_artifacts/mini_eval_generations/
```

If `POST_TRAIN_CLEAN_LOCAL=True`, the local mini-eval generation directory is
removed after upload, the same way rollout artifacts are cleaned. Pull the files
back from OSS before offline analysis if needed.

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

The AMBER image layout downloaded from the official link is supported directly:

```text
/home/liuyanlin.lyl/notebook/data/AMBER/AMBER_1.jpg
/home/liuyanlin.lyl/notebook/data/AMBER/data/query/query_all.json
```

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

The recommended source is the HuggingFace parquet dataset
`lmms-lab/MME`. The script can read that layout directly:

```bash
MME_HF_ROOT=/home/liuyanlin.lyl/notebook/data/MME_hf \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

For low-disk machines, stage the HF parquet dataset from OSS into
`/home/liuyanlin.lyl/notebook/data/MME_hf` only for the current run:

```bash
MME_HF_OSS_URI=oss://<bucket>/<path>/MME_hf \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

If no local/OSS data is present, the script attempts:

```bash
huggingface-cli download lmms-lab/MME --repo-type dataset \
  --local-dir /home/liuyanlin.lyl/notebook/data/MME_hf
```

The older official-release layout is still supported:

```bash
MME_ROOT=/path/to/MME_Benchmark_release_version \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

Converted JSON/JSONL with `image`, `question`, `answer`, and `category`
fields is also supported:

```bash
MME_JSON=/path/to/mme_perception.json \
bash res-opd/scripts/tmp/val_mme_perception.sh <merged_checkpoint_path> latest
```

Outputs for both AMBER and MME are placed under:

```text
res-opd/eval_results/<version>/<experiment>_<step>/final_hallucination/
```

When `MME_HF_ROOT` or `MME_HF_OSS_URI` is used, the script first converts the
HF parquet dataset to the existing `MME_JSON` format, then runs the normal MME
evaluator. The converted images/JSON are written under
`/home/liuyanlin.lyl/notebook/data/MME_hf_converted/` and removed after eval
unless `KEEP_BENCHMARK_DATA=True`.

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
oss://<bucket>/<path>/benchmarks/MME_hf/
```

After a manual first download, upload with:

```bash
ossutil cp -r /home/liuyanlin.lyl/notebook/data/AMBER/ \
  oss://<bucket>/<path>/benchmarks/AMBER/ -f
ossutil cp -r /home/liuyanlin.lyl/notebook/data/MME_hf/ \
  oss://<bucket>/<path>/benchmarks/MME_hf/ -f
```

ModelScope fallback is configurable but intentionally not hard-coded:

```bash
AMBER_MODELSCOPE_ID=<owner/dataset>
MME_MODELSCOPE_ID=<owner/dataset>
```

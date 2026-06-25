# Res-OPD Probe Experiments

This directory is for small verification probes that are not part of the main
training or benchmark pipeline. Probe outputs should be written under
`res-opd/probes/results/` and are ignored by git.

## Low-Resolution Teacher Object Probe

Validate COCO instance annotation structure on the full test split:

```bash
cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD
source activate vision-opd
python -u res-opd/probes/probe_lowres_teacher_objects.py \
  --validate-coco-only \
  --test-json res-opd/data/test_1000.json \
  --instances-json /home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/instances_val2017.json
```

Run a small smoke test on base-model student captions:

```bash
cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD
source activate vision-opd
python -u res-opd/probes/probe_lowres_teacher_objects.py \
  --model-path /home/liuyanlin.lyl/notebook/model/qwen/Qwen3VL-2B-Instruct \
  --eval-results <BASE_TEST1000_EVAL_RESULTS_JSONL> \
  --test-json res-opd/data/test_1000.json \
  --instances-json /home/liuyanlin.lyl/notebook/data/COCO/coco2017val/annotations/instances_val2017.json \
  --degradation-mode original \
  --student-ratio 1.0 \
  --teacher-ratio 0.75 \
  --probe-views student,teacher \
  --topk 50 \
  --topk-breaks 1,10,20,30 \
  --max-samples 20 \
  --overwrite
```

Default outputs are written to:

```text
res-opd/probes/results/lowres_teacher_objects/<run_name>/
├── coco_instance_structure_summary.json
├── lowres_teacher_object_probe.jsonl
└── lowres_teacher_object_probe_summary.json
```

`--topk-breaks` does not trigger extra model forwards. The script first stores
the requested `--topk` logprobs, then slices them into extra summary fields such
as `teacher_top1_logprob_mean`, `teacher_top10_logprob_mean`,
`teacher_top20_entropy_mean`, and `teacher_top30_mass_mean`.

Use `--probe-views student,teacher` when you need a direct high-res vs low-res
object-existence comparison. The JSONL stores yes/no forced-scoring values for
both views, and the summary reports:

- `student_state_by_label`
- `teacher_state_by_label`
- `reject_metrics_by_view`
- `student_teacher_state_pairs_by_label`
- `student_teacher_reject_quadrants`

These fields are computed during summary/merge, so existing JSONL files produced
with `--probe-views student,teacher` can be re-merged without another model
forward.

## Multi-GPU Sharding

Run one process per GPU. Each shard writes a separate JSONL and summary with a
`.shardXX-of-YY` suffix:

```bash
CUDA_VISIBLE_DEVICES=0 python -u res-opd/probes/probe_lowres_teacher_objects.py ... --num-shards 8 --shard-index 0
CUDA_VISIBLE_DEVICES=1 python -u res-opd/probes/probe_lowres_teacher_objects.py ... --num-shards 8 --shard-index 1
```

After all shards finish, merge summaries:

```bash
python -u res-opd/probes/probe_lowres_teacher_objects.py \
  --merge-jsonl-glob 'res-opd/probes/results/lowres_teacher_objects/<run_name>/lowres_teacher_object_probe.shard*-of-08.jsonl' \
  --merge-output-json res-opd/probes/results/lowres_teacher_objects/<run_name>/lowres_teacher_object_probe_summary.merged.json \
  --overwrite
```

The merge command uses `--support-margin 0.5` and `--confident-entropy-max 0.65`
by default. Pass different values during merge if the state thresholds need to
be swept.

## RKL Mention Reduction Diagnostic

After a baseline low-res object probe has been run with
`--probe-views student,teacher`, compare baseline captions against an RKL
checkpoint's captions:

```bash
python -u res-opd/probes/analyze_rkl_mention_reduction.py \
  --baseline-results res-opd/eval_results/baseline_new/Qwen3VL-2B-Instruct/train1500_test300/eval_results.jsonl \
  --model-results res-opd/eval_results/latest/full/<experiment>_global_step_39/<dataset>/eval_results.jsonl \
  --baseline-probe-jsonl-glob 'res-opd/probes/results/lowres_teacher_objects/<run_name>/lowres_teacher_object_probe*.jsonl' \
  --test-json res-opd/data/test_1000.json \
  --output-dir res-opd/eval_results/latest/full/<experiment>_global_step_39/<dataset>/mention_reduction_analysis
```

This script does not run inference. It compares object mentions in
`eval_results.jsonl` and joins baseline mentions with the per-object probe JSONL
to test whether mentions removed by RKL mainly come from:

- `both_reject`
- high caption-token entropy
- low cross-resolution support

Outputs:

```text
mention_reduction_analysis/
├── mention_reduction_summary.json
├── mention_reduction_summary.md
├── baseline_object_status.jsonl
├── model_added_objects.jsonl
└── sample_mention_diff.jsonl
```

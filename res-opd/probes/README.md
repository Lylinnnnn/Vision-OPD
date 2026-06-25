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
  --probe-views teacher \
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

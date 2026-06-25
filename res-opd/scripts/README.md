# Res-OPD Scripts

Training, evaluation, and utility scripts for Res-OPD (Regional-to-Global On-Policy Self-Distillation).

## Quick Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_res_opd_default.sh` | Training with optimized defaults | `bash res-opd/scripts/run_res_opd_default.sh` |
| `run_res_opd.sh` | Training (full config, low-level) | Called by `run_res_opd_default.sh` |
| `run_score_eval_trace.sh` | Forced scoring on eval results | `bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25` |
| `run_serial_experiments.sh` | Run multiple experiments sequentially | See script header |
| `eval_after_merge.sh` | Evaluate a merged checkpoint | `bash res-opd/scripts/eval_after_merge.sh <checkpoint>` |
| `eval_batch_from_oss.sh` | Batch eval from OSS checkpoints | See script header |
| `merge_checkpoint.sh` | Merge FSDP shards into HF model | `bash res-opd/scripts/merge_checkpoint.sh <ckpt_dir>` |
| `prepare_data.py` | Download & preprocess training data | `python res-opd/scripts/prepare_data.py --data-dir ./data` |
| `fetch_training_artifacts_from_oss.sh` | Pull traces/rollouts from OSS | See script header |
| `ckpt_upload_watcher.sh` | Background checkpoint upload to OSS | Auto-launched by training script |

---

## Training

### Default Launcher (Recommended)

```bash
bash res-opd/scripts/run_res_opd_default.sh
```

Optimized defaults for 8× H20 GPUs (conservative):

| Parameter | Default | Original | Notes |
|-----------|:-------:|:--------:|-------|
| `DATA_DATALOADER_NUM_WORKERS` | 4 | 0 | Avoids main-process blocking on the 5k/full datasets |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.8 | 0.7 | Better KV cache utilization |
| `ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU` | 8 | 1 | Better GPU utilization for logprob |
| `REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU` | 8 | 1 | Same as above |
| `ACTOR_PARAM_OFFLOAD` | False | True | H20 98GB default; set True if OOM |
| `ACTOR_OPTIMIZER_OFFLOAD` | False | True | Same |
| `REF_PARAM_OFFLOAD` | False | True | Same |

All parameters can be overridden via environment variables:

```bash
# Aggressive settings (disable offload, more workers)
ACTOR_PARAM_OFFLOAD=False \
ACTOR_OPTIMIZER_OFFLOAD=False \
REF_PARAM_OFFLOAD=False \
DATA_DATALOADER_NUM_WORKERS=4 \
TRAIN_BATCH_SIZE=64 \
bash res-opd/scripts/run_res_opd_default.sh
```

Run in tmux for long training:

```bash
tmux new-session -d -s opd_train \
  "cd /path/to/Vision-OPD && bash res-opd/scripts/run_res_opd_default.sh 2>&1 | tee res-opd/logs/train.log"
```

### Low-Level Launcher

`run_res_opd.sh` is the underlying script that accepts all configuration via environment variables. See the script header for the full parameter reference. `run_res_opd_default.sh` simply sets optimized defaults and calls it.

Selective low-res veto is off by default. Enable it only for targeted OPD
experiments:

```bash
DEGRADATION_MODE=original \
STUDENT_RATIO=1.0 \
TEACHER_RATIO=0.75 \
TEACHER_MODE=frozen \
ALPHA=1.0 \
OPD_SELECTIVE_VETO=True \
OPD_SELECTIVE_VETO_TOP_P=0.10 \
bash res-opd/scripts/run_res_opd.sh
```

`OPD_SELECTIVE_VETO_TOP_P` is a per-microbatch percentile over valid response
tokens. The implementation keeps only positive veto gaps
`student_logprob - teacher_logprob > OPD_SELECTIVE_VETO_MIN_SCORE` and selects
the largest top-p fraction. The default `0.10` is intentionally not too strict.
`OPD_SELECTIVE_VETO_NORMALIZE=True` keeps the selected-token loss on a comparable
scale to normal token-mean RKL/JSD.

For recall-safer token masking, reuse the original high-divergence token mask:

```bash
DEGRADATION_MODE=original \
STUDENT_RATIO=1.0 \
TEACHER_RATIO=0.75 \
TEACHER_MODE=frozen \
ALPHA=1.0 \
OPD_TOKEN_MASK_PCT=0.10 \
OPD_TOKEN_MASK_METRIC=student_teacher_delta \
OPD_BUCKET_METRICS=True \
OPD_BUCKET_Q_LOW=0.70 \
OPD_BUCKET_Q_HIGH=0.90 \
bash res-opd/scripts/run_res_opd_default.sh
```

`OPD_TOKEN_MASK_PCT=0.10` masks the top 10% highest-scoring valid response tokens
per sample, so it keeps about 90%; use `0.30` for a stricter keep-70% variant.
`OPD_TOKEN_MASK_METRIC=student_teacher_delta` ranks tokens by
`student_logprob - teacher_logprob`; `loss` ranks by the raw distillation loss.
Training logs only compact token-mask metrics by default:
`self_distillation/token_mask_masked_frac_valid` and
`self_distillation/token_mask_masked_tokens`. This keeps the legacy hard-mask
path usable without flooding SwanLab. Set `OPD_SELECTIVE_METRICS_VERBOSE=True`
to restore detailed token-mask bucket hit rates for debugging.

For a recall-safer RKL variant, prefer soft token weighting over hard masking:

```bash
DEGRADATION_MODE=original \
STUDENT_RATIO=1.0 \
TEACHER_RATIO=0.75 \
TEACHER_MODE=frozen \
ALPHA=1.0 \
OPD_SELECTIVE_WEIGHT=True \
OPD_SELECTIVE_WEIGHT_PROTECT=0.5 \
OPD_SELECTIVE_WEIGHT_UNCLEAR=0.5 \
OPD_SELECTIVE_WEIGHT_RISK=1.0 \
OPD_SELECTIVE_WEIGHT_OTHER=1.0 \
OPD_SELECTIVE_WEIGHT_NORMALIZE=True \
OPD_BUCKET_METRICS=True \
bash res-opd/scripts/run_res_opd_default.sh
```

`OPD_SELECTIVE_WEIGHT=True` applies a soft weight to each token's RKL/JSD loss
using the `entropy_rkl_bucket` rule:

| Bucket | Rule | Default weight | Purpose |
| --- | --- | ---: | --- |
| protect | student entropy <= q40 and raw distill loss <= q40 | 0.5 | Reduce low-res teacher pressure on low-risk tokens |
| risk | student entropy >= q75 and raw distill loss >= q75 | 1.0 | Keep full raw RKL pressure on high-risk tokens |
| unclear | student entropy >= q75 and raw distill loss <= q50 | 0.5 | Avoid over-penalizing high-entropy tokens that teacher does not clearly reject |
| other | all remaining valid tokens | 1.0 | Keep full raw RKL pressure |

The implementation normalizes weights over valid tokens by default, so this is
not just a smaller global RKL learning rate:

```text
weighted_loss_t = raw_loss_t * raw_weight_t / mean(raw_weight_valid)
```

With normalization on, tokens in `risk`/`other` can receive an effective weight
above 1.0 when many `protect`/`unclear` tokens are downweighted. This is
intentional: the average distillation strength stays stable while gradients are
shifted away from low-risk tokens.

This mode automatically enables entropy computation in `run_res_opd.sh`.
Monitor the compact SwanLab metrics:

- `self_distillation/bucket/support_*`: tokens where low-res teacher assigns at
  least as much logprob as the high-res student.
- `self_distillation/bucket/disagree_*`: tokens where the high-res student is
  more confident than the low-res teacher.
- `self_distillation/bucket/*_student_entropy_mean` and
  `self_distillation/bucket/*_teacher_entropy_mean`: whether disagreement is
  mostly coming from uncertain regions.
- `self_distillation/bucket/*_delta_mean`: mean
  `student_logprob - teacher_logprob`; larger positive values mean stronger
  cross-resolution disagreement.

- `self_distillation/selective_weight_effective_mean` should stay near 1.0 when normalization is on.
- `self_distillation/selective_weight_raw_mean` shows how much the unnormalized
  weights would shrink or amplify the average RKL/JSD loss.
- `self_distillation/selective_weight_weighted_abs_loss_over_raw` shows the
  global absolute-loss magnitude after selective weighting.
- `self_distillation/selective_weight_protect_frac`
- `self_distillation/selective_weight_unclear_frac`
- `self_distillation/selective_weight_risk_frac`
- `self_distillation/selective_weight_other_frac`
- `self_distillation/selective_weight_bucket/<bucket>_effective_weight_mean`
  confirms the actual post-normalization weight for each bucket.
- `self_distillation/selective_weight_bucket/<bucket>_raw_abs_loss_share` vs
  `<bucket>_weighted_abs_loss_share` shows whether gradient mass moved away
  from `protect/unclear` and stayed on `risk/other`.

Verbose curves are off by default. Use
`OPD_TRAIN_METRICS_VERBOSE=True` for top-k training metrics and
`OPD_SELECTIVE_METRICS_VERBOSE=True` for mild/medium/strong disagreement
thresholds, token-mask bucket hit rates, selective-veto thresholds, and
selective-weight quantile thresholds.

The intended validation is not only CHAIR/POPE: rerun
`res-opd/probes/analyze_rkl_mention_reduction.py` and check whether
`student=support|teacher=support` correct removals decrease while `both_reject`
removal remains high.

---

## Eval Trace Scoring

## Batch Eval From OSS

`eval_batch_from_oss.sh` downloads merged checkpoints from OSS, evaluates them,
and removes local model files after results are written. The default eval mode
is `chair,pope`. Optional external hallucination benchmarks are off by default
and can be enabled explicitly:

```bash
bash res-opd/scripts/eval_batch_from_oss.sh \
  --oss-names <oss_exp_name> \
  --step global_step_46 \
  --version-tag latest \
  --eval-mode chair,pope,amber,mme
```

Aliases: `coco` or `frequent` -> `chair,pope`; `final` -> `amber,mme`;
`all` -> `chair,pope,amber,mme`.

### Score with Experiment Shortcut

```bash
bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25
```

Available shortcuts:

| Shortcut | Model | Student Ratio | Teacher Ratio |
|----------|-------|:-------------:|:-------------:|
| `sr1.0-tr0.25` | Res-OPD-...-orig-sr1.0-tr0.25-a0.5-ema-e1 | 1.0 | 0.25 |
| `sr1.0-tr0.5` | Res-OPD-...-orig-sr1.0-tr0.5-a0.5-ema-e1 | 1.0 | 0.5 |
| `sr1.0-tr0.75` | Res-OPD-...-orig-sr1.0-tr0.75-a0.5-ema-e1 | 1.0 | 0.75 |
| `sr0.25-tr1.0` | Res-OPD-...-orig-sr0.25-tr1.0-a0.5-ema-e1 | 0.25 | 1.0 |
| `sr0.5-tr1.0` | Res-OPD-...-orig-sr0.5-tr1.0-a0.5-ema-e1 | 0.5 | 1.0 |
| `sr0.75-tr1.0` | Res-OPD-...-orig-sr0.75-tr1.0-a0.5-ema-e1 | 0.75 | 1.0 |

The script automatically:
- Derives model path, OSS checkpoint, eval_results, and case_analysis paths
- Downloads checkpoint from OSS (skips if already exists locally)
- Fixes cuDNN version mismatch
- Cleans up local checkpoint after scoring (`--cleanup-after`)
- Saves output trace alongside eval_results (default path)

Override defaults with extra args:

```bash
# Quick test with 50 samples
bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25 --max-samples 50 --topk 20

# Full scoring (default: all samples, topk=100, entropy enabled)
bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25
```

Run in tmux:

```bash
tmux new-session -d -s opd_score_tr025 \
  "cd /path/to/Vision-OPD && bash res-opd/scripts/run_score_eval_trace.sh sr1.0-tr0.25 2>&1 | tee res-opd/logs/score_tr0.25.log"
```

### Analyze Traces

After scoring, analyze teacher-student token-level behavior:

```bash
python res-opd/eval/analyze_opd_trace.py \
    --trace-dir <eval-results-dir>/opd_eval_trace.jsonl \
    --case-analysis <eval-results-dir>/case_analysis/all_cases_sorted.json
```

Output files (`opd_trace_summary.json`, `opd_trace_summary.md`) are saved in the same directory as the trace file by default.
The analysis includes object-level correct vs hallucinated summaries, student
entropy bins, logprob-delta quantiles, and a gate sweep for rules like
`student_entropy > e && teacher_minus_student_logp < -m`.

For the base-model low-resolution critic sanity check, use
`python res-opd/eval/run_base_trace_probe.py` on a fixed train-probe
`eval_results.jsonl`. It scores one high-resolution student view against
multiple low-resolution critic ratios and writes `base_trace_probe_summary.md`.
That summary includes both the original suppression/gate table and the
low-resolution support/reject/uncertain quadrant table.

---

## Output Directory Structure

Each experiment's artifacts are organized under `eval_results/latest/`:

```
eval_results/latest/{model_name}/{dataset}/
├── eval_results.jsonl              # Raw inference results
├── chair_metrics.json              # CHAIR hallucination metrics
├── case_analysis/                  # Badcase/goodcase analysis
│   ├── all_cases_sorted.json
│   ├── badcases.json
│   ├── goodcases.json
│   └── summary.json
├── opd_eval_trace.jsonl            # Forced scoring trace
├── opd_trace_summary.json          # Trace analysis (JSON)
├── opd_trace_summary.md            # Trace analysis (Markdown)
├── trace_analysis.json             # Training trace analysis
└── trace_analysis_with_cases.json  # Training trace + case-overlap analysis
```

Older directories named `{model_name}_{step}/{dataset}/` are still accepted as
a fallback by `run_score_eval_trace.sh`, but the current preferred layout is
`{model_name}/{dataset}/`.

---

## Environment Notes

### cuDNN Version Fix

If you encounter `cuDNN version incompatibility` errors, the launcher scripts automatically fix this by prepending the conda-bundled cuDNN to `LD_LIBRARY_PATH`. Manual fix:

```bash
export LD_LIBRARY_PATH=/path/to/conda/envs/vision-opd/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
```

### Disk Space

Checkpoints are ~4GB each. Use `--cleanup-after` in scoring scripts to auto-delete local checkpoints after use. The training script supports `POST_TRAIN_CLEAN_LOCAL=True` to clean up after training.

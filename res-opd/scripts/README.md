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
| `DATA_DATALOADER_NUM_WORKERS` | 2 | 0 | Avoids main-process blocking; increase to 4 if stable |
| `ROLLOUT_GPU_MEMORY_UTILIZATION` | 0.85 | 0.7 | Better KV cache utilization |
| `ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU` | 2 | 1 | Better GPU utilization for logprob |
| `REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU` | 2 | 1 | Same as above |
| `ACTOR_PARAM_OFFLOAD` | True | True | Keep for safety; disable when confirmed stable |
| `ACTOR_OPTIMIZER_OFFLOAD` | True | True | Same |
| `REF_PARAM_OFFLOAD` | True | True | Same |

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

---

## Eval Trace Scoring

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

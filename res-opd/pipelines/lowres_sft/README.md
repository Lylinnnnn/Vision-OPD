# Low-res SFT Pipeline

This pipeline builds a sampled-SFT baseline for Res-OPD:

1. Generate captions from degraded images at `--lowres-ratio`.
2. Build a multi-turn SFT parquet where the model input image is original resolution.
3. Train with verl SFT.
4. Merge and upload checkpoints to the same OSS layout used by `eval_batch_from_oss.sh`.

User-facing entrypoint:

```bash
bash res-opd/scripts/run_lowres_sft.sh --lowres-ratio 0.75
```

Implementation files live here; `res-opd/scripts/` should only contain launchers and operational shell utilities.

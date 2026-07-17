# Low-res SFT Pipeline

This pipeline builds a sampled-SFT baseline for Res-OPD:

1. Generate captions from degraded images at `--lowres-ratio`.
2. Build a multi-turn SFT parquet where the model input image is original resolution.
3. Train with verl SFT.
4. Merge and upload checkpoints to the same OSS layout used by `eval_batch_from_oss.sh`.

Generation caching is shared by resolution, e.g.
`res-opd/sft_data/tr0.75/lowres_generations.jsonl`, and guarded by
`lowres_generations.complete.json`. The launcher validates this marker and the
JSONL contents before skipping vLLM. Set `STRICT_GENERATION_MODEL_CACHE=True`
if the cache must also match the generating model name.

User-facing entrypoint:

```bash
bash res-opd/scripts/run_lowres_sft.sh --lowres-ratio 0.75
```

Implementation files live here; `res-opd/scripts/` should only contain launchers and operational shell utilities.

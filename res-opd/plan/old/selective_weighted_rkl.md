# Selective Weighted RKL Plan

Goal: reduce the recall drop of frozen RKL while keeping its hallucination
suppression effect.

Observation from mention-reduction analysis:

- RKL removes hallucinated objects much more often than correct objects.
- Removed objects are enriched in `both_reject`, high entropy, and low
  cross-resolution support regions.
- RKL still removes many `student=support|teacher=support` correct objects,
  which explains recall loss.

Training constraint:

- Object-level yes/no `both_reject` is an offline diagnostic, not a cheap
  training-time signal.
- Training should stay token/distribution-level and avoid POPE/object-QA
  supervision.

Implementation:

1. Add optional token-level soft weights on top of existing RKL/JSD loss.
2. Default remains unchanged.
3. First experimental mode: `entropy_rkl_bucket`.
4. Use per-microbatch quantiles over valid response tokens:
   - `protect`: low entropy and low raw distillation loss
   - `risk`: high entropy and high raw distillation loss
   - `unclear`: high entropy but low raw distillation loss
   - `other`: remaining valid tokens
5. Assign conservative default weights:
   - protect = 0.5
   - unclear = 0.5
   - risk = 1.0
   - other = 1.0
6. Normalize weights by valid-token mean so the average distillation strength is
   approximately unchanged:

```text
weighted_loss_t = raw_loss_t * raw_weight_t / mean(raw_weight_valid)
```

With normalization, `risk` and `other` tokens can receive an effective weight
above 1.0 whenever many `protect`/`unclear` tokens are downweighted. This keeps
the global RKL scale stable and shifts relative gradient mass away from
low-risk tokens.

Expected behavior:

- Preserve more low-risk / likely-correct tokens by reducing teacher pressure.
- Keep the original RKL pressure on high-risk tokens.
- Avoid confounding results with a simple global loss-scale reduction.

Primary validation:

- CHAIRi should not rise much relative to frozen RKL.
- ObjRecall should improve relative to frozen RKL.
- Mention-reduction diagnostic should show lower removal of
  `student=support|teacher=support` correct objects while keeping high removal
  rate in `both_reject`.

Default monitoring:

- Keep SwanLab compact by default.
- Track only `support` and `disagree` agreement buckets, with student/teacher
  entropy, student/teacher selected-token logprob, and
  `student_logprob - teacher_logprob` delta.
- Track selective-weight hit rates for `protect`, `unclear`, `risk`, and
  `other`, plus raw/effective mean weights.
- Track per selective-weight bucket effective weight and raw-vs-weighted
  absolute loss share. These are the main validation curves for whether the
  method actually moves gradient mass away from low-risk/unclear tokens and
  keeps it on high-risk tokens.
- Keep legacy hard token mask quiet by default: log only masked fraction and
  masked token count.
- Turn on `OPD_TRAIN_METRICS_VERBOSE=True` only when top-k train diagnostics are
  needed.
- Turn on `OPD_SELECTIVE_METRICS_VERBOSE=True` only when debugging
  mild/medium/strong disagreement thresholds, token-mask bucket hit rates, or
  selective-weight quantile thresholds.

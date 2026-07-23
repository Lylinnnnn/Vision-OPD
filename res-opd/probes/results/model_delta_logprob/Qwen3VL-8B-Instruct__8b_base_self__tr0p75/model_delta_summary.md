# Model-Delta Logprob Probe

## Purpose

This probe explains P2/P3: given base-generated captions, it compares the trained model against the base model on the original image, while keeping the low-resolution base view as the RKL gate reference.

Important convention: `RKL(after || base)`, `KL(base || after)`, and `JSD(after, base)` are all reported for the actual post-training distribution shift. `RKL(base || low-res teacher)` is the pre-training RiskMask gate signal.

## Inputs

- trace_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_base_self__tr0p75/model_delta_trace.jsonl`
- after_name: `8b_base_self`
- eval_results: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/instruct/full/Qwen3VL-8B-Instruct/train5000_test1000_original_sr1p0/eval_results.jsonl`
- after_eval_results: `None`
- records: 993
- tokens: 573475
- object_mentions: 16221

## Token Groups

| group | N | gate RKL | base NLL | base Ent. | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 16045 | 0.0351 | 0.2560 | 0.4417 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | n/a |
| hallucinated_object | 2950 | 0.0613 | 0.4391 | 0.6565 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | n/a |
| non_object_token | 550129 | 0.0342 | 0.3611 | 0.6249 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | n/a |
| unknown_token | 4341 | 0.0436 | 0.3358 | 0.5835 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | n/a |

## RiskMask-NLL Top-p vs Rest

| group | N | gate RKL | base NLL | base Ent. | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | 172043 | 0.0999 | 0.9067 | 1.3249 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | n/a |
| unselected | 401432 | 0.0064 | 0.1234 | 0.3174 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | n/a | n/a | n/a |

## Object Mention Groups

| group | N | gate RKL | base NLL | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | selected frac | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 13469 | 0.0369 | 0.2739 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a |
| hallucinated_object | 2742 | 0.0576 | 0.4444 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | n/a | n/a | n/a | n/a |

## Mechanism Counts

| mechanism | tokens | rate |
| --- | --- | --- |
| stable | 573475 | 1.0000 |

## Example Files

- examples_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_base_self__tr0p75/model_delta_examples.jsonl`

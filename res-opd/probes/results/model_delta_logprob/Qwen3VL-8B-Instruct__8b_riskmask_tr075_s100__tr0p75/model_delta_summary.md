# Model-Delta Logprob Probe

## Purpose

This probe explains P2/P3: given base-generated captions, it compares the trained model against the base model on the original image, while keeping the low-resolution base view as the RKL gate reference.

Important convention: `RKL(after || base)`, `KL(base || after)`, and `JSD(after, base)` are all reported for the actual post-training distribution shift. `RKL(base || low-res teacher)` is the pre-training RiskMask gate signal.

## Inputs

- trace_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_riskmask_tr075_s100__tr0p75/model_delta_trace.jsonl`
- after_name: `8b_riskmask_tr075_s100`
- eval_results: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/instruct/full/Qwen3VL-8B-Instruct/train5000_test1000_original_sr1p0/eval_results.jsonl`
- after_eval_results: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/instruct/full/Res-OPD-Qwen3VL-8B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-riskmask-nll-p30-b16-rn4-full5k-e1_global_step_100/train5000_test1000_original_sr1p0/eval_results.jsonl`
- records: 993
- tokens: 573475
- object_mentions: 16221

## Token Groups

| group | N | gate RKL | base NLL | base Ent. | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 16045 | 0.0351 | 0.2560 | 0.4417 | 0.0175 | 0.0042 | 0.0132 | -0.0047 | 0.9647 | 0.9072 | 0.2081 | 0.0355 | 0.0857 |
| hallucinated_object | 2950 | 0.0613 | 0.4391 | 0.6565 | 0.0292 | 0.0069 | 0.0056 | 0.0127 | 0.9481 | 0.9011 | 0.2034 | 0.0529 | 0.1478 |
| non_object_token | 550129 | 0.0342 | 0.3611 | 0.6249 | 0.0172 | 0.0041 | 0.0103 | -0.0003 | 0.9536 | 0.9171 | 0.2168 | 0.0468 | 0.1139 |
| unknown_token | 4341 | 0.0436 | 0.3358 | 0.5835 | 0.0209 | 0.0047 | 0.0151 | -0.0006 | 0.9512 | 0.9117 | 0.1995 | 0.0488 | 0.1009 |

## RiskMask-NLL Top-p vs Rest

| group | N | gate RKL | base NLL | base Ent. | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | 172043 | 0.0999 | 0.9067 | 1.3249 | 0.0453 | 0.0109 | 0.0402 | -0.0088 | 0.8600 | 0.9017 | 0.3068 | 0.1406 | 0.2171 |
| unselected | 401432 | 0.0064 | 0.1234 | 0.3174 | 0.0052 | 0.0012 | -0.0024 | 0.0032 | 0.9940 | 0.9232 | 0.1776 | 0.0063 | 0.0687 |

## Object Mention Groups

| group | N | gate RKL | base NLL | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | selected frac | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 13469 | 0.0369 | 0.2739 | 0.0185 | 0.0045 | 0.0149 | -0.0058 | n/a | n/a | n/a | n/a |
| hallucinated_object | 2742 | 0.0576 | 0.4444 | 0.0274 | 0.0065 | 0.0040 | 0.0097 | n/a | n/a | n/a | n/a |

## Base Mention Fate Under Trained Model

| fate | N | gate RKL | base NLL | Delta logp | JSD(after,base) | selected frac |
| --- | --- | --- | --- | --- | --- | --- |
| kept_correct | 13332 | 0.0366 | 0.2707 | 0.0152 | 0.0044 | n/a |
| kept_hallucinated | 2260 | 0.0482 | 0.3850 | 0.0055 | 0.0057 | n/a |
| removed_correct | 137 | 0.0729 | 0.5783 | -0.0153 | 0.0101 | n/a |
| removed_hallucinated | 482 | 0.1014 | 0.7231 | -0.0030 | 0.0102 | n/a |

## Mechanism Counts

| mechanism | tokens | rate |
| --- | --- | --- |
| stable | 334475 | 0.5832 |
| suppress_emitted | 64932 | 0.1132 |
| sharpen_same_top1 | 124089 | 0.2164 |
| boost_emitted | 23267 | 0.0406 |
| reshape_topk | 26712 | 0.0466 |

## Example Files

- examples_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_riskmask_tr075_s100__tr0p75/model_delta_examples.jsonl`

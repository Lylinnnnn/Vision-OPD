# Model-Delta Logprob Probe

## Purpose

This probe explains P2/P3: given base-generated captions, it compares the trained model against the base model on the original image, while keeping the low-resolution base view as the RKL gate reference.

Important convention: `RKL(after || base)`, `KL(base || after)`, and `JSD(after, base)` are all reported for the actual post-training distribution shift. `RKL(base || low-res teacher)` is the pre-training RiskMask gate signal.

## Inputs

- trace_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_rkl_tr075_s312__tr0p75/model_delta_trace.jsonl`
- after_name: `8b_rkl_tr075_s312`
- eval_results: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/instruct/full/Qwen3VL-8B-Instruct/train5000_test1000_original_sr1p0/eval_results.jsonl`
- after_eval_results: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/eval_results/instruct/full/Res-OPD-Qwen3VL-8B-Instruct-orig-sr1.0-tr0.75-a1.0-frozen-rkl-b16-rn4-full5k-e1_global_step_312/train5000_test1000_original_sr1p0/eval_results.jsonl`
- records: 993
- tokens: 573475
- object_mentions: 16221

## Token Groups

| group | N | gate RKL | base NLL | base Ent. | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 16045 | 0.0351 | 0.2560 | 0.4417 | 0.0233 | 0.0055 | 0.0167 | -0.0114 | 0.9579 | 0.8951 | 0.2353 | 0.0426 | 0.0830 |
| hallucinated_object | 2950 | 0.0613 | 0.4391 | 0.6565 | 0.0415 | 0.0094 | 0.0116 | 0.0044 | 0.9359 | 0.8904 | 0.2447 | 0.0654 | 0.1417 |
| non_object_token | 550129 | 0.0342 | 0.3611 | 0.6249 | 0.0228 | 0.0054 | 0.0135 | -0.0068 | 0.9474 | 0.9067 | 0.2462 | 0.0533 | 0.1157 |
| unknown_token | 4341 | 0.0436 | 0.3358 | 0.5835 | 0.0297 | 0.0066 | 0.0158 | -0.0089 | 0.9431 | 0.8980 | 0.2313 | 0.0576 | 0.1062 |

## RiskMask-NLL Top-p vs Rest

| group | N | gate RKL | base NLL | base Ent. | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| selected | 172043 | 0.0999 | 0.9067 | 1.3249 | 0.0610 | 0.0145 | 0.0488 | -0.0214 | 0.8407 | 0.8881 | 0.3319 | 0.1604 | 0.2155 |
| unselected | 401432 | 0.0064 | 0.1234 | 0.3174 | 0.0067 | 0.0015 | -0.0015 | -0.0007 | 0.9934 | 0.9139 | 0.2088 | 0.0071 | 0.0716 |

## Object Mention Groups

| group | N | gate RKL | base NLL | RKL(after,base) | JSD(after,base) | Delta logp | Delta Ent. | selected frac | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| correct_object | 13469 | 0.0369 | 0.2739 | 0.0245 | 0.0058 | 0.0176 | -0.0121 | n/a | n/a | n/a | n/a |
| hallucinated_object | 2742 | 0.0576 | 0.4444 | 0.0388 | 0.0089 | 0.0094 | 0.0001 | n/a | n/a | n/a | n/a |

## Base Mention Fate Under Trained Model

| fate | N | gate RKL | base NLL | Delta logp | JSD(after,base) | selected frac |
| --- | --- | --- | --- | --- | --- | --- |
| kept_correct | 13315 | 0.0362 | 0.2702 | 0.0174 | 0.0058 | n/a |
| kept_hallucinated | 2253 | 0.0474 | 0.3805 | 0.0107 | 0.0073 | n/a |
| removed_correct | 154 | 0.1010 | 0.5884 | 0.0332 | 0.0118 | n/a |
| removed_hallucinated | 489 | 0.1045 | 0.7390 | 0.0031 | 0.0161 | n/a |

## Mechanism Counts

| mechanism | tokens | rate |
| --- | --- | --- |
| stable | 315136 | 0.5495 |
| suppress_emitted | 65835 | 0.1148 |
| sharpen_same_top1 | 140938 | 0.2458 |
| reshape_topk | 30446 | 0.0531 |
| boost_emitted | 21120 | 0.0368 |

## Example Files

- examples_jsonl: `/home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_rkl_tr075_s312__tr0p75/model_delta_examples.jsonl`

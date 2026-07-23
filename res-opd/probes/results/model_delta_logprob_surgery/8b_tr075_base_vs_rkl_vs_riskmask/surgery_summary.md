# Model-Delta Surgery Probe

This is a post-processing analysis over `model_delta_trace.jsonl`. It is designed to explain *where* OPD training changes the base-caption distribution, and whether RiskMask behaves like a targeted edit rather than a generic 30% weakening of RKL.

Important: `offline_risk_tail` means tokens that would be selected by the RiskMask score on the base trace. For a RiskMask checkpoint this approximates the trained mask; for Frozen RKL it is only a counterfactual bin, because Frozen RKL applies RKL to all valid response tokens.

## Runs

| run | records | tokens | labeled mentions | trace |
| --- | --- | --- | --- | --- |
| base_self | 993 | 573475 | 16211 | /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_base_self__tr0p75/model_delta_trace.jsonl |
| frozen_rkl_s312 | 993 | 573475 | 16211 | /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_rkl_tr075_s312__tr0p75/model_delta_trace.jsonl |
| riskmask_p30_s100 | 993 | 573475 | 16211 | /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD/res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_riskmask_tr075_s100__tr0p75/model_delta_trace.jsonl |

## Focus Mention Buckets

These buckets directly answer the paper questions: stable correct mentions, uncertain correct mentions, high-risk hallucinations, confident hallucinations, and low-RKL uncertain hallucinations.

| run/bucket | N | share | hallu rate | removed rate | gate RKL | base NLL | base Ent. | teacher-base NLL | JSD(after,base) | Delta logp | Delta Ent. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_self/correct_lowrisk_certain | 8324 | 0.5135 | 0.0000 | 0.0000 | 0.0026 | 0.0437 | 0.1595 | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| base_self/correct_highrisk_uncertain | 2431 | 0.1500 | 0.0000 | 0.0000 | 0.1432 | 0.9255 | 1.2554 | -0.0716 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| base_self/hallucination_highrisk_uncertain | 784 | 0.0484 | 1.0000 | 0.0000 | 0.1594 | 1.1210 | 1.4083 | -0.0360 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| base_self/hallucination_highrisk_confident | 275 | 0.0170 | 1.0000 | 0.0000 | 0.0965 | 0.1502 | 0.5360 | 0.1073 | -0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| base_self/hallucination_lowrisk_uncertain | 309 | 0.0191 | 1.0000 | 0.0000 | 0.0084 | 0.7501 | 1.0979 | -0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| frozen_rkl_s312/correct_lowrisk_certain | 8324 | 0.5135 | 0.0000 | 0.0068 | 0.0026 | 0.0437 | 0.1595 | 0.0007 | 0.0009 | -0.0004 | -0.0031 | 0.1804 | 0.0009 | 0.0314 |
| frozen_rkl_s312/correct_highrisk_uncertain | 2431 | 0.1500 | 0.0000 | 0.0259 | 0.1432 | 0.9255 | 1.2554 | -0.0716 | 0.0210 | 0.0900 | -0.0465 | 0.3579 | 0.1904 | 0.1732 |
| frozen_rkl_s312/hallucination_highrisk_uncertain | 784 | 0.0484 | 1.0000 | 0.2717 | 0.1594 | 1.1210 | 1.4083 | -0.0360 | 0.0227 | 0.0564 | -0.0227 | 0.3589 | 0.1782 | 0.2079 |
| frozen_rkl_s312/hallucination_highrisk_confident | 275 | 0.0170 | 1.0000 | 0.1782 | 0.0965 | 0.1502 | 0.5360 | 0.1073 | 0.0109 | -0.0438 | 0.0643 | 0.3024 | 0.0000 | 0.3512 |
| frozen_rkl_s312/hallucination_lowrisk_uncertain | 309 | 0.0191 | 1.0000 | 0.2395 | 0.0084 | 0.7501 | 1.0979 | -0.0032 | 0.0060 | -0.0018 | -0.0133 | 0.2762 | 0.1181 | 0.2136 |
| riskmask_p30_s100/correct_lowrisk_certain | 8324 | 0.5135 | 0.0000 | 0.0061 | 0.0026 | 0.0437 | 0.1595 | 0.0007 | 0.0007 | -0.0013 | 0.0007 | 0.1500 | 0.0007 | 0.0286 |
| riskmask_p30_s100/correct_highrisk_uncertain | 2431 | 0.1500 | 0.0000 | 0.0230 | 0.1432 | 0.9255 | 1.2554 | -0.0716 | 0.0160 | 0.0801 | -0.0349 | 0.3530 | 0.1561 | 0.1817 |
| riskmask_p30_s100/hallucination_highrisk_uncertain | 784 | 0.0484 | 1.0000 | 0.2704 | 0.1594 | 1.1210 | 1.4083 | -0.0360 | 0.0166 | 0.0441 | -0.0032 | 0.3300 | 0.1495 | 0.2384 |
| riskmask_p30_s100/hallucination_highrisk_confident | 275 | 0.0170 | 1.0000 | 0.1855 | 0.0965 | 0.1502 | 0.5360 | 0.1073 | 0.0077 | -0.0417 | 0.0541 | 0.2724 | 0.0109 | 0.2906 |
| riskmask_p30_s100/hallucination_lowrisk_uncertain | 309 | 0.0191 | 1.0000 | 0.1974 | 0.0084 | 0.7501 | 1.0979 | -0.0032 | 0.0039 | -0.0136 | -0.0001 | 0.2341 | 0.0728 | 0.2314 |

## Offline Risk Tail vs Safe Bulk

Use this table to show whether training changes are localized. Do not call the Frozen RKL risk tail a training mask.

| run/bin | N | share | risk rate | hallu obj tok rate | gate RKL | base NLL | base Ent. | teacher-base NLL | JSD(after,base) | Delta logp | Delta Ent. | Top1 same | TopK Jac. | Sharpen | Reshape | Suppress |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Figure Suggestions

1. Heatmap: `token_role_x_rkl_nll`, with color = suppress rate or JSD(after,base), faceted by run. This directly shows where RKL/RiskMask performs surgery.
2. Stacked bars: focus mention buckets, with bars = kept/removed and color = correct/hallucinated. This shows why hallucinated objects decrease while correct objects are preserved.
3. Scatter/hexbin from `surgery_quadrants.csv`: x = gate RKL, y = base NLL, color = after_change_status, facet = run. This is the most intuitive plot for the RiskMask story.
4. Example panel: use `surgery_examples.jsonl`, especially `removed_hallucination_highrisk_suppressed`, `kept_correct_lowrisk_stable`, and `failure_removed_correct`.

## Writing Guidance

- The central claim should be targeted editing: RiskMask acts on high-RKL/high-NLL regions while leaving low-risk correct regions stable.
- For Frozen RKL, `offline_risk_tail` is a diagnostic bin only. Phrase it as "tokens that would be selected by RiskMask".
- If high-risk correct mentions are mostly sharpened/kept, write that RiskMask protects visually valid but uncertain mentions by not blindly suppressing every uncertain token.
- If confident hallucinations remain hard, write it as a limitation: NLL-based RiskMask is strongest when hallucination is both visually unstable and low-confidence; confidently hallucinated priors need stronger teacher-rejection or entropy-based variants.
- Use examples for the figure. Numeric means alone are too abstract for this mechanism claim.


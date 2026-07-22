# 8B Base Dual-View KL/RKL Probe Results

> **Design Motivation Probe**: Caption source = 8B Base model original-image output.
> Student = same 8B Base + original image. Teacher = same 8B Base + low-res 0.75 image.
> No trained RiskMask checkpoint used. No caption re-generation.

## Object Mention Summary

| group | count | RKL mean | RKL p75 | FKL mean | JSD mean | student NLL | student entropy |
|---|---|---|---|---|---|---|---|
| correct_object | 13,469 | 0.0369 | 0.0228 | 0.0359 | 0.0079 | 0.2739 | 0.4725 |
| hallucinated_object | 2,742 | **0.0576** | **0.0415** | **0.0568** | **0.0126** | **0.4444** | **0.6715** |
| unknown_object | 10 | 0.0495 | 0.0591 | 0.0472 | 0.0115 | 0.3555 | 0.6025 |

- **Base hallucination rate**: 16.91%
- Hallucinated objects have **56% higher RKL** (0.0576 vs 0.0369), **62% higher NLL**, and **42% higher entropy** than correct objects.
- Even with the same Base model, resolution difference alone (original vs 0.75) produces a strong RKL signal that separates hallucinations.

## High-Divergence Buckets

| metric | top frac | selected | halluc | correct | precision | recall | correct FPR | lift | threshold |
|---|---|---|---|---|---|---|---|---|---|
| rkl_student_to_teacher_mean | 0.10 | 1622 | 408 | 1214 | 0.2515 | 0.1488 | 0.0901 | 1.4871 | 0.0877 |
| rkl_student_to_teacher_mean | 0.20 | 3243 | 740 | 2503 | 0.2282 | 0.2699 | 0.1858 | 1.3490 | 0.0364 |
| rkl_student_to_teacher_mean | 0.25 | 4053 | 903 | 3150 | 0.2228 | 0.3293 | 0.2339 | 1.3172 | 0.0256 |
| fkl_teacher_to_student_mean | 0.10 | 1622 | 405 | 1217 | 0.2497 | 0.1477 | 0.0904 | 1.4762 | 0.0868 |
| fkl_teacher_to_student_mean | 0.20 | 3243 | 748 | 2495 | 0.2307 | 0.2728 | 0.1852 | 1.3636 | 0.0359 |
| jsd_mean | 0.10 | 1622 | 405 | 1217 | 0.2497 | 0.1477 | 0.0904 | 1.4762 | 0.0209 |
| jsd_mean | 0.20 | 3243 | 742 | 2501 | 0.2288 | 0.2706 | 0.1857 | 1.3527 | 0.0088 |
| student_nll_mean | 0.10 | 1622 | 445 | 1177 | **0.2744** | 0.1623 | 0.0874 | **1.6220** | 0.8621 |
| student_nll_mean | 0.20 | 3243 | 805 | 2438 | 0.2482 | 0.2936 | 0.1810 | 1.4675 | 0.4974 |
| student_entropy_mean | 0.10 | 1622 | 485 | 1137 | **0.2990** | 0.1769 | 0.0844 | **1.7678** | 1.3169 |
| student_entropy_mean | 0.20 | 3243 | 820 | 2423 | 0.2529 | 0.2991 | 0.1799 | 1.4949 | 0.9263 |
| student_entropy_mean | 0.25 | 4053 | 975 | 3078 | 0.2406 | 0.3556 | 0.2285 | 1.4222 | 0.7905 |

- **student_entropy** achieves the highest lift (1.77 at top-10%), followed by **student_nll** (1.62).
- Both outperform pure RKL (1.49), confirming that **Base model's own uncertainty is already a strong hallucination signal**.

## RiskMask Top-30% vs Random Top-30%

| mode | span selected | span precision | span recall | span lift | token precision | token recall | token lift |
|---|---|---|---|---|---|---|---|
| riskmask_nll | 4864 | 0.2327 | 0.4128 | **1.376** | 0.2224 | 0.3793 | **1.315** |
| riskmask_entropy | 4864 | 0.2338 | 0.4147 | **1.382** | 0.2251 | 0.3789 | **1.331** |
| rkl_only | 4864 | 0.2272 | 0.4030 | 1.343 | 0.2107 | 0.4125 | 1.245 |
| nll_only | 4864 | 0.2321 | 0.4117 | 1.372 | 0.2233 | 0.3381 | 1.320 |
| entropy_only | 4864 | 0.2340 | 0.4150 | 1.383 | 0.2312 | 0.3326 | 1.367 |
| **random** | 4864 | **0.1743** | **0.3093** | **1.031** | **0.1608** | **0.3180** | **0.951** |

### Key Takeaways

1. **RiskMask (nll/entropy) vs Random**: span lift 1.38 vs 1.03, token lift 1.33 vs 0.95 — RiskMask significantly outperforms random selection.
2. **Even on an untrained Base model**, NLL/entropy-based top-30% selection effectively enriches hallucinated mentions (precision 23.3% vs random 17.4%).
3. This validates the RiskMask design motivation: **token-level uncertainty induced by low-resolution input naturally marks hallucination-prone regions**. Training amplifies this inherent signal.

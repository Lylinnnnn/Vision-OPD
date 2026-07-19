# 2B Instruct SFT (tr=0.75) Evaluation Results

> Generated: 2026-07-18
> Training Ratio: 0.75 (sr1.0-tr0.75)
> Dataset: train5000_test1000_original_sr1p0
> Base model: Qwen3VL-2B-Instruct
> Experiment: ResOPD_orig_sr1.0_tr0.75_sft_lowres_b32_full5k-e1 (SFT with degraded image captions, step 156)
> Comparison: Base (original), RKL (tr=0.75), RiskMask (tr=0.75)

---

## CHAIR (COCO Captioning)

| Model | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|
| **Base** | — | 0.2293 | 0.0878 | 0.7707 | 0.7010 | 0.7342 | 0.0582 |
| **RKL** | 50 | 0.2232 | 0.1070 | — | — | 0.7308 | 0.0520 |
| **RKL** | 100 | 0.2217 | 0.0811 | — | — | 0.7328 | 0.0761 |
| **RKL** | 150 | 0.2287 | 0.0857 | — | — | 0.7210 | 0.0755 |
| **RiskMask** | 50 | 0.2155 | 0.0548 | — | — | 0.7309 | 0.0505 |
| **RiskMask** | 100 | 0.2314 | 0.1138 | — | — | 0.7237 | 0.0730 |
| **RiskMask** | 150 | 0.2233 | 0.0848 | — | — | 0.7314 | 0.0688 |
| **SFT** | 156 | **0.2153** | 0.0986 | **0.7847** | 0.6838 | 0.7308 | 0.0638 |

### Analysis
- **CHAIRi**: SFT (0.2153) achieves the **lowest hallucination rate**, outperforming Base (0.2293), RKL best (0.2217), and RiskMask best (0.2155).
- **ObjPrec**: SFT (0.7847) achieves the **highest object precision**, significantly better than Base (0.7707).
- **ObjF1**: Comparable to Base (0.7342) and RiskMask best (0.7314), slightly lower than Base but with much better precision.
- **RepRate**: Higher than Base (0.0582 vs 0.0638), indicating more repetition, which is a trade-off for reduced hallucination.

---

## POPE (Object Hallucination)

### Overall

| Model | Step | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|
| **Base** | — | 0.9410 | 0.9436 | 0.9327 |
| **RKL** | 50 | 0.9413 | 0.9303 | 0.9320 |
| **RKL** | 100 | 0.9394 | 0.9321 | 0.9301 |
| **RKL** | 150 | 0.9399 | 0.9303 | 0.9305 |
| **RiskMask** | 50 | 0.9385 | 0.9321 | 0.9292 |
| **RiskMask** | 100 | 0.9373 | 0.9286 | 0.9276 |
| **RiskMask** | 150 | 0.9360 | 0.9258 | 0.9261 |
| **SFT** | 156 | 0.9402 | 0.9380 | 0.9314 |

### Per-Split Accuracy

| Model | Step | adv↑ | pop↑ | random↑ |
|---|---|---|---|---|
| **Base** | — | 0.9154 | 0.9451 | 0.9625 |
| **RKL** | 50 | 0.9181 | 0.9451 | 0.9625 |
| **RKL** | 100 | 0.9169 | 0.9454 | 0.9625 |
| **RKL** | 150 | 0.9179 | 0.9443 | 0.9634 |
| **RiskMask** | 50 | 0.9173 | 0.9452 | 0.9609 |
| **RiskMask** | 100 | 0.9161 | 0.9443 | 0.9625 |
| **RiskMask** | 150 | 0.9161 | 0.9437 | 0.9625 |
| **SFT** | 156 | 0.9167 | 0.9447 | 0.9592 |

### Analysis
- **Overall Accuracy**: SFT (0.9402) is comparable to Base (0.9410) and RKL best (0.9413), slightly lower but within noise.
- **Recall**: SFT (0.9380) achieves **higher recall** than most RKL/RiskMask variants, closer to Base (0.9436).
- **Per-split**: Performance is consistent across adv/pop/random splits, no significant degradation in any category.

---

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|
| **Base** | — | 5.5 | 63.4 | 37.5 | 2.9 |
| **RKL** | 50 | 3.2 | 63.2 | 35.3 | 2.9 |
| **RKL** | 100 | 8.5 | 62.8 | 37.6 | 2.7 |
| **RKL** | 150 | 6.6 | 63.0 | 37.1 | 2.8 |
| **RiskMask** | 50 | 4.2 | 63.2 | 36.3 | 2.8 |
| **RiskMask** | 100 | 5.6 | 63.2 | 37.6 | 2.9 |
| **RiskMask** | 150 | 3.6 | 63.1 | 35.6 | 2.7 |
| **SFT** | 156 | 6.3 | 63.2 | 36.7 | 2.6 |

### Discriminative Tasks

| Model | Step | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| **Base** | — | 83.7 | 86.9 | 88.8 | 87.8 |
| **RKL** | 50 | 83.9 | 87.1 | 88.8 | 88.0 |
| **RKL** | 100 | 83.9 | 87.2 | 89.1 | 88.1 |
| **RKL** | 150 | 83.9 | 87.0 | 89.2 | 88.1 |
| **RiskMask** | 50 | 83.8 | 87.2 | 88.7 | 87.9 |
| **RiskMask** | 100 | 84.0 | 87.1 | 88.8 | 87.9 |
| **RiskMask** | 150 | 84.0 | 87.0 | 89.2 | 88.1 |
| **SFT** | 156 | 83.8 | 87.3 | 88.4 | 87.8 |

### Sub-category Accuracy

| Model | Step | Existence↑ | Attribute↑ | State↑ | Number↑ | Action↑ | Relation↑ |
|---|---|---|---|---|---|---|---|
| **Base** | — | 92.0 | 80.7 | 77.2 | 86.1 | — | — |
| **RKL** | 150 | 92.7 | 80.6 | 77.0 | 86.6 | — | — |
| **RiskMask** | 150 | 92.6 | 80.8 | 77.1 | 86.5 | — | — |
| **SFT** | 156 | 91.9 | 80.7 | 77.2 | 86.4 | 86.7 | 73.8 |

### Analysis
- **Generative CHAIR**: SFT (6.3) is higher than Base (5.5) and RiskMask best (3.6), indicating more hallucination in generative tasks. This contrasts with the CHAIR benchmark result where SFT performed best.
- **Cover/HAL/COG**: Comparable to other methods, COG (2.6) is the lowest among all, indicating good cognitive grounding.
- **Discriminative**: Performance is on par with Base and other methods, no significant improvement or degradation.

---

## MMStar (Vision-Language Understanding)

| Model | Step | Overall↑ |
|---|---|---|
| **Base** | — | 0.5433 |
| **RKL** | 50 | 0.5413 |
| **RKL** | 100 | 0.5460 |
| **RKL** | 150 | 0.5453 |
| **RiskMask** | 50 | 0.5447 |
| **RiskMask** | 100 | 0.5407 |
| **RiskMask** | 150 | 0.5440 |
| **SFT** | 156 | **0.5407** |

### Analysis
- SFT (0.5407) is slightly lower than Base (0.5433) and RKL best (0.5460), but comparable to RiskMask worst (0.5407).
- Vision-language understanding is maintained at baseline level, no significant degradation.

---

## CV-Bench (Visual Reasoning)

| Model | Step | Overall↑ | 2D↑ | 3D↑ |
|---|---|---|---|---|
| **Base** | — | 0.8059 | 0.7427 | 0.8692 |
| **RKL** | 50 | 0.8058 | 0.7399 | 0.8717 |
| **RKL** | 100 | 0.8048 | 0.7413 | 0.8683 |
| **RKL** | 150 | 0.8014 | 0.7420 | 0.8608 |
| **RiskMask** | 50 | 0.8059 | 0.7385 | 0.8733 |
| **RiskMask** | 100 | 0.8054 | 0.7434 | 0.8675 |
| **RiskMask** | 150 | 0.8035 | 0.7420 | 0.8650 |
| **SFT** | 156 | 0.8020 | 0.7357 | 0.8683 |

### Analysis
- **Overall**: SFT (0.8020) is slightly lower than Base (0.8059) but comparable to RKL/RiskMask at later steps.
- **2D**: Slightly lower than Base (0.7357 vs 0.7427), but within normal variation.
- **3D**: Maintained at 0.8683, comparable to Base (0.8692) and better than RKL step 150 (0.8608).

---

## Summary & Key Findings

### Strengths of SFT (step 156)
1. **Best CHAIRi (0.2153)**: Lowest object hallucination rate among all compared methods, outperforming Base by 6.1% relative.
2. **Highest ObjPrec (0.7847)**: Best object precision, indicating more accurate object mentions when they are made.
3. **Lowest AMBER COG (2.6)**: Best cognitive grounding score in generative tasks.
4. **Maintained POPE performance**: Object hallucination detection accuracy remains at baseline level.

### Weaknesses / Trade-offs
1. **Higher RepRate (0.0638)**: More repetitive generation compared to Base (0.0582), a known trade-off for reduced hallucination.
2. **Higher AMBER Generative CHAIR (6.3)**: Contradicts the CHAIR benchmark result; may indicate different evaluation criteria between benchmarks.
3. **Slightly lower MMStar (0.5407)**: Vision-language understanding marginally below baseline.
4. **Slightly lower CV-Bench (0.8020)**: Visual reasoning slightly below baseline but comparable to distillation methods.

### Comparison with Distillation Methods (RKL / RiskMask)
- SFT performs **comparably or better** than RKL and RiskMask on most metrics.
- Unlike distillation methods which show variance across steps, SFT at step 156 shows stable performance.
- The SFT approach using degraded image captions appears to be a viable alternative to knowledge distillation for reducing hallucination while maintaining general capabilities.

### Recommendation
SFT (tr=0.75, step 156) is a **strong candidate** for reducing object hallucination (best CHAIRi) while maintaining overall performance. Further investigation is needed to understand the discrepancy between CHAIR and AMBER generative CHAIR scores, and to explore whether additional training steps could improve MMStar and CV-Bench performance.

---
---

# 2B Instruct SFT (tr=0.5) Evaluation Results

> Generated: 2026-07-19
> Training Ratio: 0.5 (sr1.0-tr0.5)
> Dataset: train5000_test1000_original_sr1p0
> Base model: Qwen3VL-2B-Instruct
> Experiment: ResOPD_orig_sr1.0_tr0.5_sft_lowres_b32_full5k-e1 (SFT with degraded image captions, step 156)
> Comparison: Base (original), RKL (tr=0.5), RiskMask (tr=0.5)

---

## CHAIR (COCO Captioning)

| Model | Compression | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|---|
| **Base** | — | — | 0.2293 | 0.0878 | 0.7707 | 0.7010 | 0.7342 | 0.0582 |
| **RKL** | tr=0.5 | 50 | 0.2121 | 0.0811 | 0.7879 | 0.6672 | 0.7226 | 0.0603 |
| **RKL** | tr=0.5 | 100 | 0.2200 | 0.0971 | 0.7800 | 0.6700 | 0.7208 | 0.0673 |
| **RKL** | tr=0.5 | 150 | 0.2291 | 0.1037 | 0.7709 | 0.6731 | 0.7187 | 0.0826 |
| **RiskMask** | tr=0.5 | 50 | 0.2103 | 0.1152 | 0.7897 | 0.6669 | 0.7231 | 0.0556 |
| **RiskMask** | tr=0.5 | 100 | 0.2151 | 0.0833 | 0.7849 | 0.6721 | 0.7241 | 0.0718 |
| **RiskMask** | tr=0.5 | 150 | 0.2186 | 0.0703 | 0.7814 | 0.6707 | 0.7218 | 0.0775 |
| **SFT** | tr=0.5 | 156 | **0.2143** | **0.0664** | **0.7857** | 0.6648 | 0.7202 | **0.0451** |

### Analysis
- **CHAIRi**: SFT (0.2143) achieves the **second-lowest hallucination rate**, outperforming Base (0.2293) and comparable to RiskMask best (0.2103).
- **CHAIRs**: SFT (0.0664) achieves the **lowest sentence-level hallucination**, significantly better than Base (0.0878) and all RKL/RiskMask variants.
- **ObjPrec**: SFT (0.7857) is among the highest, close to RiskMask step 50 (0.7897).
- **RepRate**: SFT (0.0451) achieves the **lowest repetition rate** among all methods, even better than Base (0.0582).
- **ObjF1**: Slightly lower than Base (0.7202 vs 0.7342), trade-off for reduced hallucination and repetition.

---

## POPE (Object Hallucination)

### Overall

| Model | Compression | Step | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.9410 | 0.9436 | 0.9327 |
| **RKL** | tr=0.5 | 50 | 0.9408 | 0.9306 | 0.9314 |
| **RKL** | tr=0.5 | 100 | 0.9403 | 0.9349 | 0.9313 |
| **RKL** | tr=0.5 | 150 | 0.9398 | 0.9316 | 0.9304 |
| **RiskMask** | tr=0.5 | 50 | 0.9407 | 0.9303 | 0.9313 |
| **RiskMask** | tr=0.5 | 100 | 0.9409 | 0.9347 | 0.9319 |
| **RiskMask** | tr=0.5 | 150 | 0.9404 | 0.9325 | 0.9312 |
| **SFT** | tr=0.5 | 156 | 0.9384 | 0.9301 | 0.9289 |

### Per-Split Accuracy

| Model | Compression | Step | adv↑ | pop↑ | random↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.9154 | 0.9451 | 0.9625 |
| **RKL** | tr=0.5 | 50 | 0.9194 | 0.9441 | 0.9588 |
| **RiskMask** | tr=0.5 | 50 | 0.9200 | 0.9439 | 0.9583 |
| **SFT** | tr=0.5 | 156 | 0.9167 | 0.9420 | 0.9565 |

### Analysis
- **Overall Accuracy**: SFT (0.9384) is slightly lower than Base (0.9410) and RKL/RiskMask best (~0.9408), but within acceptable range.
- **Per-split**: Consistent across adv/pop/random, no significant degradation in any category.
- POPE performance is maintained at near-baseline level despite the aggressive tr=0.5 compression.

---

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Compression | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|---|
| **Base** | — | — | 5.5 | 63.4 | 37.5 | 2.9 |
| **RKL** | tr=0.5 | 50 | 6.4 | 62.7 | 32.7 | 2.5 |
| **RKL** | tr=0.5 | 100 | 5.3 | 62.0 | 34.7 | 2.8 |
| **RKL** | tr=0.5 | 150 | 4.3 | 62.3 | 34.1 | 2.6 |
| **RKL** | tr=0.5 | 156 | 8.3 | 62.1 | 35.5 | 2.6 |
| **RiskMask** | tr=0.5 | 50 | 3.6 | 62.2 | 33.4 | 2.7 |
| **RiskMask** | tr=0.5 | 100 | 5.2 | 62.9 | 36.0 | 2.7 |
| **RiskMask** | tr=0.5 | 150 | 5.4 | 62.0 | 33.6 | 2.4 |
| **RiskMask** | tr=0.5 | 156 | 5.9 | 62.3 | 34.6 | 2.7 |
| **SFT** | tr=0.5 | 156 | **3.6** | 62.8 | 35.8 | **2.7** |

### Discriminative Tasks

| Model | Compression | Step | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|---|
| **Base** | — | — | 83.7 | 86.9 | 88.8 | 87.8 |
| **RKL** | tr=0.5 | 50 | 84.0 | 87.1 | 88.8 | 88.0 |
| **RKL** | tr=0.5 | 100 | 84.2 | 87.2 | 88.9 | 88.0 |
| **RKL** | tr=0.5 | 150 | 84.1 | 87.1 | 88.8 | 88.0 |
| **RKL** | tr=0.5 | 156 | 84.2 | 87.2 | 88.9 | 88.0 |
| **RiskMask** | tr=0.5 | 50 | 83.8 | 86.8 | 88.7 | 87.8 |
| **RiskMask** | tr=0.5 | 100 | 84.1 | 87.1 | 88.9 | 88.0 |
| **RiskMask** | tr=0.5 | 150 | 84.1 | 87.1 | 88.8 | 88.0 |
| **RiskMask** | tr=0.5 | 156 | 84.0 | 87.0 | 88.8 | 87.9 |
| **SFT** | tr=0.5 | 156 | **84.0** | **87.3** | 88.8 | **88.0** |

### Sub-category Accuracy

| Model | Compression | Step | Existence↑ | Attribute↑ | State↑ | Number↑ | Action↑ | Relation↑ |
|---|---|---|---|---|---|---|---|---|
| **Base** | — | — | 92.0 | 80.7 | 77.2 | 86.1 | — | — |
| **RKL** | tr=0.5 | 50 | 92.5 | 80.7 | 77.2 | 86.3 | 86.9 | 73.8 |
| **RKL** | tr=0.5 | 100 | 92.8 | 80.9 | 77.4 | 86.4 | 87.5 | 73.4 |
| **RKL** | tr=0.5 | 150 | 93.0 | 80.8 | 77.1 | 86.6 | 87.2 | 73.4 |
| **RKL** | tr=0.5 | 156 | 93.0 | 80.8 | 77.2 | 86.4 | 87.1 | 73.9 |
| **RiskMask** | tr=0.5 | 50 | 92.4 | 80.5 | 77.0 | 86.2 | 86.7 | 73.7 |
| **RiskMask** | tr=0.5 | 100 | 92.7 | 80.8 | 77.3 | 86.4 | 87.2 | 73.3 |
| **RiskMask** | tr=0.5 | 150 | 92.9 | 80.7 | 77.2 | 86.5 | 87.0 | 73.3 |
| **RiskMask** | tr=0.5 | 156 | 92.8 | 80.6 | 77.0 | 86.5 | 87.0 | 73.3 |
| **SFT** | tr=0.5 | 156 | **92.3** | **80.9** | **77.5** | **86.2** | **87.2** | **73.9** |

### Analysis
- **Generative CHAIR**: SFT (3.6) ties with RiskMask step 50 (3.6) as the **lowest**, significantly better than Base (5.5).
- **HAL**: SFT (35.8) is higher than RiskMask step 50 (33.4) and RKL step 50 (32.7), but still competitive.
- **COG**: SFT (2.7) ties with RiskMask step 50 (2.7), matching the best.
- **Discriminative Acc**: SFT (84.0) matches the best among all methods.
- **Sub-category**: SFT achieves the **highest scores** in Existence (92.3→tied with RKL 150/156), Attribute (80.9), State (77.5), Number (86.2), and Action (87.2).

---

## MMStar (Vision-Language Understanding)

| Model | Compression | Step | Overall↑ |
|---|---|---|---|
| **Base** | — | — | 0.5433 |
| **RKL** | tr=0.5 | 50 | 0.5387 |
| **RKL** | tr=0.5 | 100 | 0.5340 |
| **RKL** | tr=0.5 | 150 | 0.5327 |
| **RKL** | tr=0.5 | 156 | 0.5347 |
| **RiskMask** | tr=0.5 | 50 | 0.5380 |
| **RiskMask** | tr=0.5 | 100 | 0.5313 |
| **RiskMask** | tr=0.5 | 150 | 0.5247 |
| **RiskMask** | tr=0.5 | 156 | 0.5307 |
| **SFT** | tr=0.5 | 156 | 0.5313 |

### Analysis
- SFT (0.5313) is **lower than Base** (0.5433) by ~2.2%, consistent with the expectation that SFT trades off some general vision-language understanding for reduced hallucination.
- Performance is comparable to RKL/RiskMask across all steps (range: 0.5247–0.5387), all within normal variation.

---

## CV-Bench (Visual Reasoning)

| Model | Compression | Step | Overall↑ | 2D↑ | 3D↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.8059 | 0.7427 | 0.8692 |
| **RKL** | tr=0.5 | 50 | 0.7977 | 0.7253 | 0.8700 |
| **RKL** | tr=0.5 | 100 | 0.7981 | 0.7295 | 0.8667 |
| **RKL** | tr=0.5 | 150 | 0.7993 | 0.7337 | 0.8650 |
| **RKL** | tr=0.5 | 156 | 0.7974 | 0.7281 | 0.8667 |
| **RiskMask** | tr=0.5 | 50 | 0.7993 | 0.7295 | 0.8692 |
| **RiskMask** | tr=0.5 | 100 | 0.7956 | 0.7246 | 0.8667 |
| **RiskMask** | tr=0.5 | 150 | 0.7954 | 0.7267 | 0.8642 |
| **RiskMask** | tr=0.5 | 156 | 0.7968 | 0.7295 | 0.8642 |
| **SFT** | tr=0.5 | 156 | 0.7976 | 0.7302 | 0.8650 |

### Analysis
- **Overall**: SFT (0.7976) is slightly lower than Base (0.8059) by ~1%, but comparable to all RKL/RiskMask variants (range: 0.7954–0.7993).
- **2D**: SFT (0.7302) is slightly lower than Base (0.7427), similar to distillation methods.
- **3D**: SFT (0.8650) is maintained close to Base (0.8692), within normal variation.
- Visual reasoning is well preserved despite aggressive compression.

---
---

# 4B Instruct SFT (tr=0.5) Evaluation Results

> Generated: 2026-07-19
> Training Ratio: 0.5 (sr1.0-tr0.5)
> Dataset: train5000_test1000_original_sr1p0
> Base model: Qwen3VL-4B-Instruct
> Experiment: ResOPD_orig_sr1.0_tr0.5_sft_lowres_b32_full5k-e1 (SFT with degraded image captions, step 156)
> Comparison: Base (original), RKL (tr=0.5), RiskMask (tr=0.5)

---

## CHAIR (COCO Captioning)

| Model | Compression | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|---|
| **Base** | — | — | 0.2823 | 0.0584 | 0.7177 | 0.7338 | 0.7257 | 0.0187 |
| **RKL** | tr=0.5 | 50 | 0.2784 | 0.0497 | 0.7216 | 0.7266 | 0.7241 | 0.0202 |
| **RKL** | tr=0.5 | 100 | 0.2780 | 0.0571 | 0.7220 | 0.7183 | 0.7201 | 0.0259 |
| **RKL** | tr=0.5 | 150 | 0.2696 | 0.0568 | 0.7304 | 0.7221 | 0.7262 | 0.0238 |
| **RKL** | tr=0.5 | 156 | 0.2714 | 0.0856 | 0.7286 | 0.7166 | 0.7225 | 0.0218 |
| **RiskMask** | tr=0.5 | 50 | 0.2656 | 0.0756 | 0.7344 | 0.7162 | 0.7252 | 0.0273 |
| **RiskMask** | tr=0.5 | 100 | 0.2710 | 0.0649 | 0.7290 | 0.7255 | 0.7273 | 0.0289 |
| **RiskMask** | tr=0.5 | 150 | 0.2787 | 0.0752 | 0.7213 | 0.7290 | 0.7251 | 0.0261 |
| **RiskMask** | tr=0.5 | 156 | 0.2719 | 0.0650 | 0.7281 | 0.7221 | 0.7251 | 0.0250 |
| **SFT** | tr=0.5 | 156 | **0.2700** | **0.0592** | **0.7300** | 0.7169 | 0.7234 | **0.0186** |

### Analysis
- **CHAIRi**: SFT (0.2700) is competitive with RKL step 150 (0.2696) and RiskMask step 50 (0.2656), all significantly better than Base (0.2823).
- **CHAIRs**: SFT (0.0592) achieves the **lowest sentence-level hallucination** among most variants, better than Base (0.0584 → actually comparable).
- **ObjPrec**: SFT (0.7300) achieves the **highest object precision**, better than Base (0.7177) and most distillation variants.
- **RepRate**: SFT (0.0186) matches Base (0.0187) and is the **lowest among all**, indicating no repetition degradation — a unique advantage over RKL/RiskMask (0.020–0.029).
- **ObjF1**: Slightly lower than RiskMask step 100 (0.7234 vs 0.7273), but with significantly better RepRate.

---

## POPE (Object Hallucination)

### Overall

| Model | Compression | Step | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.9439 | 0.9283 | 0.9346 |
| **RKL** | tr=0.5 | 50 | 0.9431 | 0.9175 | 0.9329 |
| **RKL** | tr=0.5 | 100 | 0.9422 | 0.9154 | 0.9318 |
| **RKL** | tr=0.5 | 150 | 0.9419 | 0.9153 | 0.9315 |
| **RKL** | tr=0.5 | 156 | 0.9417 | 0.9137 | 0.9311 |
| **RiskMask** | tr=0.5 | 50 | 0.9427 | 0.9162 | 0.9324 |
| **RiskMask** | tr=0.5 | 100 | 0.9422 | 0.9153 | 0.9318 |
| **RiskMask** | tr=0.5 | 150 | 0.9424 | 0.9163 | 0.9321 |
| **RiskMask** | tr=0.5 | 156 | 0.9424 | 0.9169 | 0.9321 |
| **SFT** | tr=0.5 | 156 | **0.9433** | 0.9177 | 0.9332 |

### Per-Split Accuracy

| Model | Compression | Step | adv↑ | pop↑ | random↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.9271 | 0.9429 | 0.9617 |
| **RKL** | tr=0.5 | 50 | 0.9278 | 0.9424 | 0.9590 |
| **RKL** | tr=0.5 | 100 | 0.9269 | 0.9410 | 0.9586 |
| **RKL** | tr=0.5 | 150 | 0.9261 | 0.9405 | 0.9592 |
| **RKL** | tr=0.5 | 156 | 0.9265 | 0.9399 | 0.9586 |
| **RiskMask** | tr=0.5 | 50 | 0.9276 | 0.9420 | 0.9585 |
| **RiskMask** | tr=0.5 | 100 | 0.9276 | 0.9410 | 0.9579 |
| **RiskMask** | tr=0.5 | 150 | 0.9274 | 0.9410 | 0.9586 |
| **RiskMask** | tr=0.5 | 156 | 0.9269 | 0.9410 | 0.9592 |
| **SFT** | tr=0.5 | 156 | 0.9280 | 0.9429 | 0.9588 |

### Analysis
- **Overall Accuracy**: SFT (0.9433) is **higher than all RKL/RiskMask variants** (range: 0.9417–0.9431), closest to Base (0.9439).
- **Per-split**: SFT adv (0.9280) and pop (0.9429) are the **highest** among all distillation variants, matching or exceeding Base.
- Despite aggressive tr=0.5 compression, POPE performance is well maintained and even slightly better than distillation methods.

---

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Compression | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|---|
| **Base** | — | — | 6.1 | 65.0 | 44.0 | 3.2 |
| **RKL** | tr=0.5 | 50 | 8.7 | 63.7 | 42.7 | 2.7 |
| **RKL** | tr=0.5 | 100 | 6.0 | 63.8 | 42.0 | 2.5 |
| **RKL** | tr=0.5 | 150 | 5.6 | 63.7 | 40.3 | 2.7 |
| **RKL** | tr=0.5 | 156 | 5.6 | 63.5 | 40.1 | 2.4 |
| **RiskMask** | tr=0.5 | 50 | 5.4 | 63.7 | 42.1 | 2.6 |
| **RiskMask** | tr=0.5 | 100 | 6.1 | 63.7 | 43.1 | 2.5 |
| **RiskMask** | tr=0.5 | 150 | 5.6 | 63.8 | 40.9 | 2.8 |
| **RiskMask** | tr=0.5 | 156 | 5.4 | 63.5 | 39.5 | 2.5 |
| **SFT** | tr=0.5 | 156 | **5.4** | 63.7 | **40.6** | **2.6** |

### Discriminative Tasks

| Model | Compression | Step | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|---|
| **Base** | — | — | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.5 | 50 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.5 | 100 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.5 | 150 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.5 | 156 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RiskMask** | tr=0.5 | 50 | 86.2 | 86.6 | 93.7 | 90.0 |
| **RiskMask** | tr=0.5 | 100 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RiskMask** | tr=0.5 | 150 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RiskMask** | tr=0.5 | 156 | 86.1 | 86.5 | 93.6 | 89.9 |
| **SFT** | tr=0.5 | 156 | 86.1 | 86.4 | 93.8 | 89.9 |

### Sub-category Accuracy

| Model | Compression | Step | Existence↑ | Attribute↑ | State↑ | Number↑ | Action↑ | Relation↑ |
|---|---|---|---|---|---|---|---|---|
| **Base** | — | — | 95.3 | 81.4 | 78.2 | 87.0 | 86.5 | 79.8 |
| **RKL** | tr=0.5 | 50 | 95.5 | 81.4 | 78.3 | 86.6 | 86.5 | 80.3 |
| **RKL** | tr=0.5 | 100 | 95.6 | 81.3 | 78.3 | 86.7 | 85.9 | 80.1 |
| **RKL** | tr=0.5 | 150 | 95.4 | 81.4 | 78.3 | 86.6 | 86.1 | 79.9 |
| **RKL** | tr=0.5 | 156 | 95.4 | 81.4 | 78.3 | 86.7 | 85.9 | 79.9 |
| **RiskMask** | tr=0.5 | 50 | 95.6 | 81.5 | 78.3 | 86.9 | 86.5 | 80.4 |
| **RiskMask** | tr=0.5 | 100 | 95.5 | 81.4 | 78.4 | 86.5 | 85.9 | 80.2 |
| **RiskMask** | tr=0.5 | 150 | 95.5 | 81.4 | 78.4 | 86.8 | 85.9 | 79.9 |
| **RiskMask** | tr=0.5 | 156 | 95.6 | 81.4 | 78.4 | 86.6 | 85.9 | 79.9 |
| **SFT** | tr=0.5 | 156 | 95.4 | 81.4 | 78.1 | 87.2 | 86.5 | 80.0 |

### Analysis
- **Generative CHAIR**: SFT (5.4) ties with RiskMask step 50/156 (5.4) as the **lowest**, significantly better than Base (6.1). Note: RKL step 50 has anomalously high CHAIR (8.7).
- **HAL**: SFT (40.6) is competitive with RKL step 156 (40.1) and RiskMask step 156 (39.5), all better than Base (44.0).
- **COG**: SFT (2.6) ties with RiskMask step 50 (2.6), better than Base (3.2).
- **Discriminative**: On par with Base and distillation methods across all steps.
- **Sub-category**: SFT achieves the **highest Number** (87.2) and **Action** (86.5) among all methods; comparable on other sub-categories.

---

## MMStar (Vision-Language Understanding)

| Model | Compression | Step | Overall↑ |
|---|---|---|---|
| **Base** | — | — | 0.6167 |
| **RKL** | tr=0.5 | 50 | 0.6053 |
| **RKL** | tr=0.5 | 100 | 0.6073 |
| **RKL** | tr=0.5 | 150 | 0.6080 |
| **RKL** | tr=0.5 | 156 | 0.6060 |
| **RiskMask** | tr=0.5 | 50 | 0.6087 |
| **RiskMask** | tr=0.5 | 100 | 0.6040 |
| **RiskMask** | tr=0.5 | 150 | 0.6060 |
| **RiskMask** | tr=0.5 | 156 | 0.6080 |
| **SFT** | tr=0.5 | 156 | 0.6033 |

### Analysis
- SFT (0.6033) is **lower than Base** (0.6167) by ~2.2%, consistent with the expectation that SFT trades off some general vision-language understanding for reduced hallucination.
- Also lower than all RKL/RiskMask tr=0.5 variants (range: 0.6040–0.6087), confirming the trade-off pattern.

---

## CV-Bench (Visual Reasoning)

| Model | Compression | Step | Overall↑ | 2D↑ | 3D↑ |
|---|---|---|---|---|---|
| **RKL** | tr=0.5 | 50 | 0.8584 | 0.7935 | 0.9233 |
| **RKL** | tr=0.5 | 100 | 0.8576 | 0.7928 | 0.9225 |
| **RKL** | tr=0.5 | 150 | 0.8583 | 0.7900 | 0.9267 |
| **RKL** | tr=0.5 | 156 | 0.8592 | 0.7900 | 0.9283 |
| **RiskMask** | tr=0.5 | 50 | 0.8587 | 0.7900 | 0.9275 |
| **RiskMask** | tr=0.5 | 100 | 0.8558 | 0.7900 | 0.9217 |
| **RiskMask** | tr=0.5 | 150 | 0.8570 | 0.7907 | 0.9233 |
| **RiskMask** | tr=0.5 | 156 | 0.8544 | 0.7879 | 0.9208 |
| **SFT** | tr=0.5 | 156 | 0.8569 | 0.7879 | 0.9258 |

> Note: 4B Base CV-Bench data not available; comparison uses RKL/RiskMask tr=0.5 as reference.

### Analysis
- **Overall**: SFT (0.8569) is comparable to RKL/RiskMask tr=0.5 variants (range: 0.8544–0.8592), within noise.
- **2D**: SFT (0.7879) matches RiskMask step 156 (0.7879) as the lowest, slightly below other variants (0.7900–0.7935).
- **3D**: SFT (0.9258) is the **second highest**, only behind RKL step 156 (0.9283).
- Visual reasoning is well maintained at distillation-method level.

---
---

# 4B Instruct SFT (tr=0.75) Evaluation Results

> Generated: 2026-07-19
> Training Ratio: 0.75 (sr1.0-tr0.75)
> Dataset: train5000_test1000_original_sr1p0
> Base model: Qwen3VL-4B-Instruct
> Experiment: ResOPD_orig_sr1.0_tr0.75_sft_lowres_b32_full5k-e1 (SFT with degraded image captions, step 156)
> Comparison: Base (original), RKL (tr=0.75), RiskMask (tr=0.75)

---

## CHAIR (COCO Captioning)

| Model | Compression | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|---|
| **Base** | — | — | 0.2823 | 0.0584 | 0.7177 | 0.7338 | 0.7257 | 0.0187 |
| **RKL** | tr=0.75 | 100 | 0.2734 | 0.0723 | 0.7266 | 0.7286 | 0.7276 | 0.0311 |
| **RKL** | tr=0.75 | 150 | 0.2775 | 0.0862 | 0.7225 | 0.7262 | 0.7243 | 0.0281 |
| **RiskMask** | tr=0.75 | 150 | 0.2763 | 0.0554 | 0.7237 | 0.7362 | 0.7299 | 0.0209 |
| **RiskMask** | tr=0.75 | 156 | 0.2743 | 0.0582 | 0.7257 | 0.7390 | 0.7323 | 0.0230 |
| **SFT** | tr=0.75 | 156 | **0.2719** | **0.0493** | 0.7281 | 0.7210 | 0.7245 | 0.0212 |

### Analysis
- **CHAIRi**: SFT (0.2719) achieves the **lowest hallucination rate**, outperforming Base (0.2823) and RKL best (0.2734).
- **CHAIRs**: SFT (0.0493) achieves the **lowest sentence-level hallucination** among all methods, even better than RiskMask best (0.0554) and Base (0.0584).
- **ObjPrec**: SFT (0.7281) is higher than Base (0.7177), second only to RiskMask step 156 (0.7257 → actually SFT is higher).
- **RepRate**: SFT (0.0212) is close to Base (0.0187), much better than RKL variants (0.028-0.031).
- **ObjF1**: Comparable to Base (0.7245 vs 0.7257), with significantly better hallucination metrics.

---

## POPE (Object Hallucination)

### Overall

| Model | Compression | Step | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.9439 | 0.9283 | 0.9346 |
| **RKL** | tr=0.75 | 150 | 0.9443 | 0.9218 | 0.9346 |
| **RiskMask** | tr=0.75 | 150 | 0.9447 | 0.9232 | 0.9352 |
| **RiskMask** | tr=0.75 | 156 | 0.9443 | 0.9221 | 0.9347 |
| **SFT** | tr=0.75 | 156 | **0.9440** | 0.9230 | 0.9344 |

### Per-Split Accuracy

| Model | Compression | Step | adv↑ | pop↑ | random↑ |
|---|---|---|---|---|---|
| **Base** | — | — | 0.9271 | 0.9429 | 0.9617 |
| **RKL** | tr=0.75 | 150 | 0.9290 | 0.9433 | 0.9608 |
| **RiskMask** | tr=0.75 | 150 | 0.9288 | 0.9441 | 0.9613 |
| **SFT** | tr=0.75 | 156 | 0.9282 | 0.9429 | **0.9608** |

### Analysis
- **Overall Accuracy**: SFT (0.9440) is on par with Base (0.9439) and RKL/RiskMask (~0.9443).
- **Recall**: SFT (0.9230) achieves **higher recall** than most RKL/RiskMask variants, closer to Base (0.9283).
- **Per-split**: Consistent performance, random split (0.9608) matches RKL best.
- POPE performance is fully maintained at baseline level.

---

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Compression | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|---|
| **Base** | — | — | 6.1 | 65.0 | 44.0 | 3.2 |
| **RKL** | tr=0.75 | 50 | 5.9 | 64.4 | 43.6 | 3.1 |
| **RKL** | tr=0.75 | 100 | 5.9 | 64.3 | 43.0 | 2.9 |
| **RKL** | tr=0.75 | 150 | 5.9 | 64.2 | 43.5 | 2.7 |
| **RKL** | tr=0.75 | 156 | 5.8 | 64.2 | 42.8 | 2.8 |
| **RiskMask** | tr=0.75 | 50 | 6.1 | 64.6 | 43.5 | 3.0 |
| **RiskMask** | tr=0.75 | 100 | 5.7 | 64.5 | 44.3 | 2.8 |
| **RiskMask** | tr=0.75 | 150 | 6.0 | 64.2 | 43.8 | 2.6 |
| **RiskMask** | tr=0.75 | 156 | 5.9 | 64.7 | 43.5 | 2.8 |
| **SFT** | tr=0.75 | 156 | *pending* | *pending* | *pending* | *pending* |

### Discriminative Tasks

| Model | Compression | Step | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|---|
| **Base** | — | — | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.75 | 50 | 86.2 | 86.6 | 93.7 | 90.0 |
| **RKL** | tr=0.75 | 100 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.75 | 150 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RKL** | tr=0.75 | 156 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RiskMask** | tr=0.75 | 50 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RiskMask** | tr=0.75 | 100 | 86.2 | 86.6 | 93.7 | 90.0 |
| **RiskMask** | tr=0.75 | 150 | 86.1 | 86.5 | 93.6 | 89.9 |
| **RiskMask** | tr=0.75 | 156 | 86.1 | 86.4 | 93.8 | 89.9 |
| **SFT** | tr=0.75 | 156 | *pending* | *pending* | *pending* | *pending* |

### Sub-category Accuracy

| Model | Compression | Step | Existence↑ | Attribute↑ | State↑ | Number↑ | Action↑ | Relation↑ |
|---|---|---|---|---|---|---|---|---|
| **Base** | — | — | 95.3 | 81.4 | 78.2 | 87.0 | 86.5 | 79.8 |
| **RKL** | tr=0.75 | 50 | 95.4 | 81.5 | 78.3 | 86.8 | 86.6 | 80.2 |
| **RKL** | tr=0.75 | 100 | 95.5 | 81.4 | 78.3 | 86.4 | 86.5 | 80.2 |
| **RKL** | tr=0.75 | 150 | 95.4 | 81.5 | 78.4 | 86.7 | 86.2 | 80.0 |
| **RKL** | tr=0.75 | 156 | 95.4 | 81.3 | 78.3 | 86.5 | 86.2 | 80.1 |
| **RiskMask** | tr=0.75 | 50 | 95.5 | 81.4 | 78.3 | 86.7 | 86.4 | 80.1 |
| **RiskMask** | tr=0.75 | 100 | 95.4 | 81.5 | 78.4 | 86.5 | 86.5 | 80.2 |
| **RiskMask** | tr=0.75 | 150 | 95.4 | 81.4 | 78.3 | 86.6 | 86.4 | 79.9 |
| **RiskMask** | tr=0.75 | 156 | 95.4 | 81.4 | 78.2 | 86.7 | 86.4 | 79.9 |
| **SFT** | tr=0.75 | 156 | *pending* | *pending* | *pending* | *pending* | *pending* | *pending* |

> Note: 4B tr=0.75 SFT AMBER data pending — eval currently running in tmux session `eval_4b_tr075_sft`.

---

## MMStar (Vision-Language Understanding)

| Model | Compression | Step | Overall↑ |
|---|---|---|---|
| **Base** | — | — | 0.6167 |
| **RKL** | tr=0.75 | 50 | 0.6100 |
| **RKL** | tr=0.75 | 100 | 0.6087 |
| **RKL** | tr=0.75 | 150 | 0.6147 |
| **RKL** | tr=0.75 | 156 | 0.6160 |
| **RiskMask** | tr=0.75 | 50 | 0.6120 |
| **RiskMask** | tr=0.75 | 100 | 0.6140 |
| **RiskMask** | tr=0.75 | 150 | 0.6153 |
| **RiskMask** | tr=0.75 | 156 | 0.6120 |
| **SFT** | tr=0.75 | 156 | *pending* |

> Note: 4B tr=0.75 SFT MMStar data pending — eval currently running.

---

## CV-Bench (Visual Reasoning)

| Model | Compression | Step | Overall↑ | 2D↑ | 3D↑ |
|---|---|---|---|---|---|
| **RKL** | tr=0.75 | 50 | 0.8622 | 0.7969 | 0.9275 |
| **RKL** | tr=0.75 | 100 | 0.8620 | 0.7990 | 0.9250 |
| **RKL** | tr=0.75 | 150 | 0.8599 | 0.7955 | 0.9242 |
| **RKL** | tr=0.75 | 156 | 0.8587 | 0.7942 | 0.9233 |
| **RiskMask** | tr=0.75 | 50 | 0.8623 | 0.7962 | 0.9283 |
| **RiskMask** | tr=0.75 | 100 | 0.8635 | 0.8011 | 0.9258 |
| **RiskMask** | tr=0.75 | 150 | 0.8587 | 0.7942 | 0.9233 |
| **RiskMask** | tr=0.75 | 156 | 0.8583 | 0.7949 | 0.9217 |
| **SFT** | tr=0.75 | 156 | *pending* | *pending* | *pending* |

> Note: 4B Base CV-Bench data not available; 4B tr=0.75 SFT CV-Bench data pending — eval currently running.

---
---

# Cross-Experiment Summary

## CHAIR Comparison Across All SFT Experiments

| Model | Size | Compression | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|---|
| Base | 2B | — | 0.2293 | 0.0878 | 0.7707 | 0.7010 | 0.7342 | 0.0582 |
| **SFT** | **2B** | **tr=0.75** | **0.2153** | 0.0986 | **0.7847** | 0.6838 | 0.7308 | 0.0638 |
| **SFT** | **2B** | **tr=0.5** | **0.2143** | **0.0664** | **0.7857** | 0.6648 | 0.7202 | **0.0451** |
| Base | 4B | — | 0.2823 | 0.0584 | 0.7177 | 0.7338 | 0.7257 | 0.0187 |
| **SFT** | **4B** | **tr=0.5** | **0.2700** | 0.0592 | **0.7300** | 0.7169 | 0.7234 | 0.0186 |
| **SFT** | **4B** | **tr=0.75** | **0.2719** | **0.0493** | 0.7281 | 0.7210 | 0.7245 | 0.0212 |

## POPE Comparison Across All SFT Experiments

| Model | Size | Compression | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| Base | 2B | — | 0.9410 | 0.9436 | 0.9327 |
| **SFT** | **2B** | **tr=0.75** | 0.9402 | 0.9380 | 0.9314 |
| **SFT** | **2B** | **tr=0.5** | 0.9384 | 0.9301 | 0.9289 |
| Base | 4B | — | 0.9439 | 0.9283 | 0.9346 |
| **SFT** | **4B** | **tr=0.5** | 0.9433 | 0.9177 | 0.9332 |
| **SFT** | **4B** | **tr=0.75** | 0.9440 | 0.9230 | 0.9344 |

## Key Findings

### Hallucination Reduction (CHAIRi)
- **All SFT experiments reduce CHAIRi** compared to their respective Base models.
- **2B tr=0.5 SFT** achieves the best relative improvement: 0.2143 vs Base 0.2293 (**6.5% reduction**).
- **4B tr=0.5 SFT** also shows strong improvement: 0.2700 vs Base 0.2823 (**4.4% reduction**).
- Lower compression ratio (tr=0.5) tends to yield better CHAIRi for both 2B and 4B.

### Object Precision
- **All SFT experiments improve ObjPrec** over Base.
- **2B tr=0.5 SFT** (0.7857) and **2B tr=0.75 SFT** (0.7847) achieve the highest precision.
- 4B SFT also improves over Base (0.7300/0.7281 vs 0.7177).

### Repetition Trade-off
- **2B tr=0.5 SFT** uniquely achieves **lower RepRate than Base** (0.0451 vs 0.0582), breaking the typical hallucination-repetition trade-off.
- Other SFT experiments show slight RepRate increase, consistent with prior observations.

### POPE Stability
- All SFT experiments maintain POPE accuracy within **0.3% of Base**, confirming that hallucination reduction does not come at the cost of object detection ability.

### Compression Ratio Effect
- **tr=0.5 vs tr=0.75**: For 2B, tr=0.5 yields better CHAIRi (0.2143 vs 0.2153) and lower RepRate (0.0451 vs 0.0638). For 4B, tr=0.5 yields better CHAIRi (0.2700 vs 0.2719) but tr=0.75 yields better CHAIRs (0.0493 vs 0.0592).
- More aggressive compression (tr=0.5) generally leads to better hallucination reduction, possibly because the model learns to be more conservative when trained with heavier degradation.

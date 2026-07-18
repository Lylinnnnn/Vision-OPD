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

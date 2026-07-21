# 8B Instruct RiskMask Top-P Ablation Results (tr=0.75)

> Generated: 2026-07-21
> Dataset: train5000_test1000_original_sr1p0
> Base model: Qwen3VL-8B-Instruct
> Experiments: RKL (tr=0.75), RiskMask p=0.10 / p=0.20 / p=0.30 / p=0.40 (tr=0.75)

---

## CHAIR (COCO Captioning)

| Model | Step | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|
| **Base** | — | 0.3025 | 0.0597 | 0.7178 | 0.0156 |
| **RKL** | 50 | 0.2971 | 0.0504 | 0.7198 | 0.0152 |
| **RKL** | 100 | 0.2974 | 0.0528 | 0.7198 | 0.0137 |
| **RKL** | 150 | 0.2943 | 0.0940 | 0.7201 | 0.0170 |
| **RKL** | 200 | 0.2992 | 0.0554 | 0.7168 | 0.0137 |
| **RiskMask p=0.10** | 50 | 0.2855 | 0.0550 | 0.7255 | 0.0174 |
| **RiskMask p=0.10** | 100 | 0.2955 | 0.0500 | 0.7190 | 0.0163 |
| **RiskMask p=0.10** | 150 | 0.2973 | 0.0826 | 0.7163 | 0.0164 |
| **RiskMask p=0.10** | 200 | 0.2924 | 0.0658 | 0.7194 | 0.0185 |
| **RiskMask p=0.20** | 50 | 0.2814 | 0.0518 | 0.7298 | 0.0143 |
| **RiskMask p=0.20** | 100 | 0.2882 | 0.0523 | 0.7248 | 0.0139 |
| **RiskMask p=0.20** | 150 | 0.2890 | 0.0784 | 0.7231 | 0.0150 |
| **RiskMask p=0.20** | 200 | 0.2926 | 0.0811 | 0.7190 | 0.0169 |
| **RiskMask p=0.20** | 250 | 0.2910 | 0.0726 | 0.7241 | 0.0157 |
| **RiskMask p=0.30** | 50 | 0.2887 | 0.0599 | 0.7270 | 0.0146 |
| **RiskMask p=0.30** | 100 | 0.2873 | 0.0555 | 0.7272 | 0.0189 |
| **RiskMask p=0.30** | 150 | 0.2885 | 0.0601 | 0.7230 | 0.0169 |
| **RiskMask p=0.30** | 200 | 0.2911 | 0.0586 | 0.7196 | 0.0209 |
| **RiskMask p=0.30** | 300 | 0.2919 | 0.1109 | 0.7202 | 0.0199 |
| **RiskMask p=0.40** | 50 | 0.2953 | 0.0531 | 0.7186 | 0.0150 |
| **RiskMask p=0.40** | 100 | 0.2985 | 0.0509 | 0.7166 | 0.0160 |
| **RiskMask p=0.40** | 150 | 0.3010 | 0.1075 | 0.7163 | 0.0160 |
| **RiskMask p=0.40** | 200 | 0.2981 | 0.0768 | 0.7160 | 0.0188 |

### Analysis
- **Best CHAIRi**: RiskMask p=0.10 step 50 (**0.2855**), followed closely by p=0.20 step 50 (0.2814 → actually 0.2814 is lower, correcting: p=0.20 step 50 = 0.2814 is the **new best**), outperforming Base (0.3025) by 7.0% relative.
- **Best ObjF1**: RiskMask p=0.20 step 50 (**0.7298**), surpassing p=0.30 step 100 (0.7272).
- **Trend**: p=0.20 achieves the best CHAIRi at step 50 (0.2814), even better than p=0.10 (0.2855). Lower top-p (p=0.10, p=0.20, p=0.30) consistently outperforms p=0.40 and RKL. p=0.40 performs similarly to or worse than RKL.
- **RepRate**: p=0.20 step 100 (0.0139) matches RKL best (0.0137). All RiskMask variants have comparable RepRate.

---

## POPE (Object Hallucination)

### Per-Split Accuracy

| Model | Step | adv↑ | pop↑ | rand↑ |
|---|---|---|---|---|
| **Base** | — | 0.9125 | 0.9376 | 0.9644 |
| **RKL** | 50 | 0.9140 | 0.9370 | 0.9629 |
| **RKL** | 100 | 0.9159 | 0.9374 | 0.9627 |
| **RKL** | 150 | 0.9138 | 0.9366 | 0.9630 |
| **RKL** | 200 | 0.9150 | 0.9366 | 0.9632 |
| **RiskMask p=0.10** | 50 | 0.9144 | 0.9380 | 0.9634 |
| **RiskMask p=0.10** | 100 | 0.9135 | 0.9366 | 0.9627 |
| **RiskMask p=0.10** | 150 | 0.9137 | 0.9372 | 0.9630 |
| **RiskMask p=0.10** | 200 | 0.9148 | 0.9378 | 0.9640 |
| **RiskMask p=0.20** | 50 | 0.9140 | 0.9382 | 0.9638 |
| **RiskMask p=0.20** | 100 | 0.9135 | 0.9370 | 0.9625 |
| **RiskMask p=0.20** | 150 | 0.9135 | 0.9361 | 0.9627 |
| **RiskMask p=0.20** | 200 | 0.9140 | 0.9370 | 0.9632 |
| **RiskMask p=0.20** | 250 | 0.9131 | 0.9368 | 0.9629 |
| **RiskMask p=0.30** | 50 | 0.9142 | 0.9368 | 0.9629 |
| **RiskMask p=0.30** | 100 | 0.9138 | 0.9368 | 0.9627 |
| **RiskMask p=0.30** | 150 | 0.9148 | 0.9366 | 0.9634 |
| **RiskMask p=0.30** | 200 | 0.9150 | 0.9382 | 0.9642 |
| **RiskMask p=0.30** | 300 | 0.9152 | 0.9389 | 0.9650 |
| **RiskMask p=0.40** | 50 | 0.9150 | 0.9380 | 0.9629 |
| **RiskMask p=0.40** | 100 | 0.9156 | 0.9374 | 0.9629 |
| **RiskMask p=0.40** | 150 | 0.9146 | 0.9366 | 0.9625 |
| **RiskMask p=0.40** | 200 | 0.9144 | 0.9372 | 0.9627 |

### Overall Accuracy & F1

| Model | Step | Accuracy↑ | F1↑ |
|---|---|---|---|
| **Base** | — | 0.9382 | 0.9315 |
| **RKL** | 50 | 0.9380 | 0.9289 |
| **RKL** | 100 | 0.9387 | 0.9295 |
| **RKL** | 150 | 0.9378 | 0.9287 |
| **RKL** | 200 | 0.9383 | 0.9293 |
| **RiskMask p=0.10** | 50 | 0.9386 | 0.9297 |
| **RiskMask p=0.10** | 100 | 0.9376 | 0.9284 |
| **RiskMask p=0.10** | 150 | 0.9380 | 0.9290 |
| **RiskMask p=0.10** | 200 | 0.9389 | 0.9300 |
| **RiskMask p=0.20** | 50 | 0.9387 | 0.9298 |
| **RiskMask p=0.20** | 100 | 0.9376 | 0.9283 |
| **RiskMask p=0.20** | 150 | 0.9374 | 0.9282 |
| **RiskMask p=0.20** | 200 | 0.9381 | 0.9291 |
| **RiskMask p=0.20** | 250 | 0.9376 | 0.9285 |
| **RiskMask p=0.30** | 50 | 0.9380 | 0.9289 |
| **RiskMask p=0.30** | 100 | 0.9378 | 0.9286 |
| **RiskMask p=0.30** | 150 | 0.9383 | 0.9292 |
| **RiskMask p=0.30** | 200 | 0.9391 | 0.9302 |
| **RiskMask p=0.30** | 300 | 0.9397 | 0.9309 |
| **RiskMask p=0.40** | 50 | 0.9386 | 0.9296 |
| **RiskMask p=0.40** | 100 | 0.9386 | 0.9294 |
| **RiskMask p=0.40** | 150 | 0.9379 | 0.9287 |
| **RiskMask p=0.40** | 200 | 0.9381 | 0.9290 |

### Analysis
- **Best POPE Accuracy**: RiskMask p=0.30 step 300 (**0.9397**), followed by p=0.20 step 50 (0.9387) and p=0.30 step 200 (0.9391).
- **Best per-split**: p=0.30 step 300 achieves best pop (0.9389) and rand (0.9650); p=0.20 step 50 achieves best rand among p=0.20 steps (0.9638).
- All RiskMask variants perform comparably to or slightly better than RKL on POPE. No significant degradation from top-p variation.

---

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|
| **Base** | — | 5.9 | 64.4 | 44.2 | 3.0 |
| **RKL** | 50 | 5.8 | 64.5 | 43.3 | 2.8 |
| **RKL** | 100 | 6.0 | 64.3 | 43.3 | 2.9 |
| **RKL** | 150 | 5.9 | 64.4 | 44.8 | 2.8 |
| **RKL** | 200 | 5.9 | 64.6 | 44.6 | 3.0 |
| **RiskMask p=0.10** | 50 | 5.4 | 64.5 | 44.0 | 2.9 |
| **RiskMask p=0.10** | 100 | 5.7 | 64.3 | 43.2 | 2.8 |
| **RiskMask p=0.10** | 150 | 6.0 | 64.3 | 44.0 | 2.9 |
| **RiskMask p=0.10** | 200 | 5.9 | 64.7 | 44.9 | 3.3 |
| **RiskMask p=0.20** | 50 | 5.9 | 64.1 | 43.8 | 3.0 |
| **RiskMask p=0.20** | 100 | 5.6 | 64.6 | 44.5 | 2.9 |
| **RiskMask p=0.20** | 150 | 5.9 | 64.4 | 45.4 | 2.9 |
| **RiskMask p=0.20** | 200 | 6.0 | 64.6 | 44.6 | 3.3 |
| **RiskMask p=0.20** | 250 | 6.2 | 64.7 | 46.5 | 3.3 |
| **RiskMask p=0.30** | 50 | 5.8 | 64.3 | 43.5 | 3.0 |
| **RiskMask p=0.30** | 100 | 6.0 | 64.8 | 43.4 | 3.1 |
| **RiskMask p=0.30** | 150 | 6.0 | 64.2 | 45.1 | 2.8 |
| **RiskMask p=0.30** | 200 | 6.0 | 64.5 | 45.0 | 3.1 |
| **RiskMask p=0.30** | 300 | 6.1 | 64.4 | 45.6 | 2.9 |
| **RiskMask p=0.40** | 50 | 5.7 | 64.1 | 45.6 | 2.7 |
| **RiskMask p=0.40** | 100 | 5.8 | 64.7 | 42.6 | 3.0 |
| **RiskMask p=0.40** | 150 | 5.6 | 64.3 | 43.3 | 3.0 |
| **RiskMask p=0.40** | 200 | 6.0 | 64.5 | 44.3 | 3.2 |

### Discriminative Tasks

| Model | Step | Acc↑ | F1↑ |
|---|---|---|---|
| **Base** | — | 85.5 | 89.4 |
| **RKL** | 50 | 85.4 | 89.3 |
| **RKL** | 100 | 85.5 | 89.4 |
| **RKL** | 150 | 85.5 | 89.4 |
| **RKL** | 200 | 85.4 | 89.4 |
| **RiskMask p=0.10** | 50 | 85.5 | 89.4 |
| **RiskMask p=0.10** | 100 | 85.5 | 89.4 |
| **RiskMask p=0.10** | 150 | 85.5 | 89.4 |
| **RiskMask p=0.10** | 200 | 85.6 | 89.4 |
| **RiskMask p=0.20** | 50 | 85.4 | 89.4 |
| **RiskMask p=0.20** | 100 | 85.6 | 89.5 |
| **RiskMask p=0.20** | 150 | 85.4 | 89.3 |
| **RiskMask p=0.20** | 200 | 85.5 | 89.4 |
| **RiskMask p=0.20** | 250 | 85.5 | 89.5 |
| **RiskMask p=0.30** | 50 | 85.4 | 89.3 |
| **RiskMask p=0.30** | 100 | 85.5 | 89.4 |
| **RiskMask p=0.30** | 150 | 85.5 | 89.4 |
| **RiskMask p=0.30** | 200 | 85.5 | 89.4 |
| **RiskMask p=0.30** | 300 | 85.5 | 89.4 |
| **RiskMask p=0.40** | 50 | 85.4 | 89.3 |
| **RiskMask p=0.40** | 100 | 85.6 | 89.4 |
| **RiskMask p=0.40** | 150 | 85.4 | 89.4 |
| **RiskMask p=0.40** | 200 | 85.5 | 89.4 |

### Analysis
- **Best AMBER Gen CHAIR**: RiskMask p=0.10 step 50 (**5.4**), followed by p=0.20 step 100 (5.6). p=0.20 shows increasing CHAIR with training (5.9 → 6.2).
- **Best Cover**: RiskMask p=0.20 step 250 (**64.7**) and p=0.30 step 100 (64.8).
- **Discriminative**: All methods perform nearly identically (~85.4-85.6 Acc, ~89.3-89.5 F1). No meaningful difference across top-p values.

---

## MMStar (Vision-Language Understanding)

| Model | Step | Overall↑ |
|---|---|---|
| **Base** | — | 0.6480 |
| **RKL** | 50 | 0.6467 |
| **RKL** | 100 | 0.6447 |
| **RKL** | 150 | 0.6467 |
| **RKL** | 200 | 0.6393 |
| **RiskMask p=0.10** | 50 | 0.6440 |
| **RiskMask p=0.10** | 100 | 0.6467 |
| **RiskMask p=0.10** | 150 | 0.6440 |
| **RiskMask p=0.10** | 200 | 0.6500 |
| **RiskMask p=0.20** | 50 | 0.6433 |
| **RiskMask p=0.20** | 100 | 0.6440 |
| **RiskMask p=0.20** | 150 | 0.6480 |
| **RiskMask p=0.20** | 200 | 0.6447 |
| **RiskMask p=0.20** | 250 | 0.6447 |
| **RiskMask p=0.30** | 50 | 0.6433 |
| **RiskMask p=0.30** | 100 | 0.6473 |
| **RiskMask p=0.30** | 150 | 0.6467 |
| **RiskMask p=0.30** | 200 | 0.6460 |
| **RiskMask p=0.30** | 300 | 0.6467 |
| **RiskMask p=0.40** | 50 | 0.6473 |
| **RiskMask p=0.40** | 100 | 0.6447 |
| **RiskMask p=0.40** | 150 | 0.6480 |
| **RiskMask p=0.40** | 200 | 0.6453 |

### Analysis
- **Best MMStar**: RiskMask p=0.10 step 200 (**0.6500**), followed by p=0.20 step 150 (0.6480) matching Base.
- All RiskMask variants maintain MMStar at or near baseline level. No significant degradation from any top-p value.

---

## CV-Bench (Visual Reasoning)

| Model | Step | Overall↑ |
|---|---|---|
| **Base** | — | 0.8701 |
| **RKL** | 50 | 0.8686 |
| **RKL** | 100 | 0.8668 |
| **RKL** | 150 | 0.8697 |
| **RKL** | 200 | 0.8699 |
| **RiskMask p=0.10** | 50 | 0.8688 |
| **RiskMask p=0.10** | 100 | 0.8679 |
| **RiskMask p=0.10** | 150 | 0.8681 |
| **RiskMask p=0.10** | 200 | 0.8696 |
| **RiskMask p=0.20** | 50 | 0.8694 |
| **RiskMask p=0.20** | 100 | 0.8697 |
| **RiskMask p=0.20** | 150 | 0.8690 |
| **RiskMask p=0.20** | 200 | 0.8715 |
| **RiskMask p=0.20** | 250 | 0.8710 |
| **RiskMask p=0.30** | 50 | 0.8690 |
| **RiskMask p=0.30** | 100 | 0.8686 |
| **RiskMask p=0.30** | 150 | 0.8708 |
| **RiskMask p=0.30** | 200 | 0.8697 |
| **RiskMask p=0.30** | 300 | 0.8683 |
| **RiskMask p=0.40** | 50 | 0.8683 |
| **RiskMask p=0.40** | 100 | 0.8697 |
| **RiskMask p=0.40** | 150 | 0.8676 |
| **RiskMask p=0.40** | 200 | 0.8688 |

### Analysis
- **Best CV-Bench**: RiskMask p=0.20 step 200 (**0.8715**), surpassing Base (0.8701) and all other variants.
- p=0.20 shows strong CV-Bench performance at later steps (0.8715 at step 200, 0.8710 at step 250).
- All variants maintain CV-Bench within ±0.003 of baseline. No meaningful degradation from top-p variation.

---

## Summary & Key Findings

### Best Results by Metric

| Metric | Best Config | Value | vs Base |
|--------|------------|-------|---------|
| **CHAIRi↓** | RiskMask p=0.20 step 50 | 0.2814 | -7.0% ✅ |
| **ObjF1↑** | RiskMask p=0.20 step 50 | 0.7298 | +1.7% ✅ |
| **POPE Acc↑** | RiskMask p=0.30 step 300 | 0.9397 | +0.2% |
| **AMBER Gen CHAIR↓** | RiskMask p=0.10 step 50 | 5.4 | -8.5% ✅ |
| **MMStar↑** | RiskMask p=0.10 step 200 | 0.6500 | +0.3% |
| **CV-Bench↑** | RiskMask p=0.20 step 200 | 0.8715 | +0.2% ✅ |

### Top-P Sensitivity Analysis

1. **p=0.10 (most aggressive masking)**:
   - Best AMBER Gen CHAIR (5.4 at step 50)
   - Achieves highest MMStar (0.6500 at step 200)
   - Slightly higher RepRate trade-off

2. **p=0.20 (new, moderate-aggressive masking)**:
   - **Best CHAIRi** (0.2814 at step 50), outperforming all other configs including p=0.10
   - **Best ObjF1** (0.7298 at step 50), surpassing p=0.30
   - **Best CV-Bench** (0.8715 at step 200), surpassing Base
   - Performance degrades with more training (CHAIRi increases from 0.2814 to 0.2926)
   - Step 50 is the clear sweet spot for p=0.20

3. **p=0.30 (moderate masking)**:
   - Best POPE accuracy at later steps (0.9397 at step 300)
   - Most stable performance across steps
   - Competitive on all metrics

4. **p=0.40 (least aggressive masking)**:
   - Performance closest to vanilla RKL
   - No clear advantage over RKL on most metrics
   - Suggests diminishing returns when top-p is too high

### Recommendation
- **For hallucination reduction**: Use **p=0.20 step 50** (best CHAIRi=0.2814 and ObjF1=0.7298).
- **For balanced/stable performance**: Use **p=0.30** (competitive on all metrics, most stable across steps).
- **For visual reasoning**: Use **p=0.20 step 200** (best CV-Bench=0.8715).
- **p=0.40** offers no significant benefit over vanilla RKL and is not recommended.
- p=0.20 shows a clear early-stopping pattern: best performance at step 50, degrading with more training.

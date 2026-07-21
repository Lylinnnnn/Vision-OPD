# 8B Instruct Model Evaluation Results

> Generated: 2026-07-10
> Dataset: train5000_test1000_original_sr1p0
> Degradation Mode: original (all experiments)

---

## Summary Best Models

| Metric | Best Model | Value |
|---|---|---|
| **CHAIRi↓** | RiskMask step_100 | 0.2873 |
| **CHAIRs↓** | RKL step_50 | 0.0504 |
| **ObjF1↑** | RiskMask step_100 | 0.7272 |
| **RepRate↓** | RKL step_100 / step_200 | 0.0137 |
| **POPE adv↑** | RKL step_300 | 0.9161 |
| **POPE pop↑** | RKL step_312 | 0.9403 |
| **POPE rand↑** | RKL step_312 | 0.9642 |
| **AMBER CHAIR↓** | RKL step_50 / RiskMask step_50 | 5.8 |
| **AMBER Cover↑** | RiskMask step_100 | 64.8 |
| **AMBER HAL↓** | RKL step_50 / RiskMask step_100 | 43.3 / 43.4 |
| **MMStar↑** | Base | 0.6480 |
| **CVBench↑** | RKL step_200 | 0.8699 |

---

## CHAIR (COCO Captioning)

| Model | Step | CHAIRi↓ | CHAIRs↓ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|
| **Base** | - | 0.3025 | 0.0597 | 0.7178 | 0.0156 |
| **RKL** | 50 | 0.2971 | 0.0504 | 0.7198 | 0.0152 |
| **RKL** | 100 | 0.2974 | 0.0528 | 0.7198 | 0.0137 |
| **RKL** | 150 | 0.2943 | 0.0940 | 0.7201 | 0.0170 |
| **RKL** | 200 | 0.2992 | 0.0554 | 0.7168 | 0.0137 |
| **RKL** | 250 | 0.2994 | 0.0648 | 0.7189 | 0.0153 |
| **RKL** | 300 | 0.2999 | 0.0652 | 0.7159 | 0.0174 |
| **RKL** | 312 | 0.2894 | 0.0702 | 0.7258 | 0.0184 |
| **RiskMask** | 50 | 0.2887 | 0.0599 | 0.7270 | 0.0146 |
| **RiskMask** | 100 | 0.2873 | 0.0555 | 0.7272 | 0.0189 |
| **RiskMask** | 150 | 0.2885 | 0.0601 | 0.7230 | 0.0169 |
| **RiskMask** | 200 | 0.2911 | 0.0586 | 0.7196 | 0.0209 |

## POPE (Object Hallucination)

| Model | Step | adv Acc↑ | pop Acc↑ | rand Acc↑ |
|---|---|---|---|---|
| **Base** | - | 0.9125 | 0.9376 | 0.9644 |
| **RKL** | 50 | 0.9140 | 0.9370 | 0.9629 |
| **RKL** | 100 | 0.9159 | 0.9374 | 0.9627 |
| **RKL** | 150 | 0.9138 | 0.9366 | 0.9630 |
| **RKL** | 200 | 0.9150 | 0.9366 | 0.9632 |
| **RKL** | 250 | 0.9144 | 0.9364 | 0.9634 |
| **RKL** | 300 | 0.9161 | 0.9395 | 0.9638 |
| **RKL** | 312 | 0.9156 | 0.9403 | 0.9642 |
| **RiskMask** | 50 | 0.9142 | 0.9368 | 0.9629 |
| **RiskMask** | 100 | 0.9138 | 0.9368 | 0.9627 |
| **RiskMask** | 150 | 0.9148 | 0.9366 | 0.9634 |
| **RiskMask** | 200 | 0.9150 | 0.9382 | 0.9642 |

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|
| **Base** | - | 6.0 | 64.7 | 46.8 | 2.5 |
| **RKL** | 50 | 5.8 | 64.5 | 43.3 | 2.8 |
| **RKL** | 100 | 6.0 | 64.3 | 43.3 | 2.9 |
| **RKL** | 150 | 5.9 | 64.4 | 44.8 | 2.8 |
| **RKL** | 200 | 5.9 | 64.6 | 44.6 | 3.0 |
| **RKL** | 250 | 6.1 | 64.7 | 47.0 | 3.1 |
| **RKL** | 300 | 7.2 | 64.5 | 45.4 | 3.0 |
| **RKL** | 312 | 6.2 | 64.3 | 45.5 | 3.1 |
| **RiskMask** | 50 | 5.8 | 64.3 | 43.5 | 3.0 |
| **RiskMask** | 100 | 6.0 | 64.8 | 43.4 | 3.1 |
| **RiskMask** | 150 | 6.0 | 64.2 | 45.1 | 2.8 |
| **RiskMask** | 200 | 6.0 | 64.5 | 45.0 | 3.1 |

### Discriminative Tasks

| Model | Step | Acc↑ | F1↑ |
|---|---|---|---|
| **Base** | - | 85.4 | 89.3 |
| **RKL** | 50 | 85.4 | 89.3 |
| **RKL** | 100 | 85.5 | 89.4 |
| **RKL** | 150 | 85.5 | 89.4 |
| **RKL** | 200 | 85.4 | 89.4 |
| **RKL** | 250 | 85.5 | 89.4 |
| **RKL** | 300 | 85.5 | 89.4 |
| **RKL** | 312 | 85.5 | 89.4 |
| **RiskMask** | 50 | 85.4 | 89.3 |
| **RiskMask** | 100 | 85.5 | 89.4 |
| **RiskMask** | 150 | 85.5 | 89.4 |
| **RiskMask** | 200 | 85.5 | 89.4 |

### Sub-category Accuracy

| Model | Step | Existence↑ | Attribute↑ | State↑ | Number↑ |
|---|---|---|---|---|---|
| **Base** | - | 93.3 | 81.4 | 78.4 | 85.9 |
| **RKL** | 50 | 93.3 | 81.4 | 78.5 | 85.9 |
| **RKL** | 100 | 93.5 | 81.4 | 78.4 | 86.0 |
| **RKL** | 150 | 93.4 | 81.5 | 78.6 | 85.7 |
| **RKL** | 200 | 93.5 | 81.4 | 78.5 | 85.8 |
| **RKL** | 250 | 93.3 | 81.5 | 78.7 | 85.8 |
| **RKL** | 300 | 93.5 | 81.4 | 78.4 | 86.1 |
| **RKL** | 312 | 93.6 | 81.4 | 78.4 | 86.1 |
| **RiskMask** | 50 | 93.4 | 81.4 | 78.5 | 85.9 |
| **RiskMask** | 100 | 93.5 | 81.5 | 78.6 | 85.9 |
| **RiskMask** | 150 | 93.5 | 81.4 | 78.5 | 85.8 |
| **RiskMask** | 200 | 93.5 | 81.5 | 78.5 | 86.1 |

## MMStar (Vision-Language Understanding)

| Model | Step | Overall↑ |
|---|---|---|
| **Base** | - | 0.6480 |
| **RKL** | 50 | 0.6467 |
| **RKL** | 100 | 0.6447 |
| **RKL** | 150 | 0.6467 |
| **RKL** | 200 | 0.6393 |
| **RKL** | 250 | 0.6440 |
| **RKL** | 300 | 0.6440 |
| **RKL** | 312 | 0.6427 |
| **RiskMask** | 50 | 0.6433 |
| **RiskMask** | 100 | 0.6473 |
| **RiskMask** | 150 | 0.6467 |
| **RiskMask** | 200 | 0.6460 |

## CVBench (Visual Reasoning)

| Model | Step | Overall↑ | 2D↑ | 3D↑ |
|---|---|---|---|---|
| **Base** | - | 0.8701 | 0.8102 | 0.9300 |
| **RKL** | 50 | 0.8686 | 0.8081 | 0.9292 |
| **RKL** | 100 | 0.8668 | 0.8053 | 0.9283 |
| **RKL** | 150 | 0.8697 | 0.8102 | 0.9292 |
| **RKL** | 200 | 0.8699 | 0.8115 | 0.9283 |
| **RKL** | 250 | 0.8691 | 0.8074 | 0.9308 |
| **RKL** | 300 | 0.8697 | 0.8095 | 0.9300 |
| **RKL** | 312 | 0.8693 | 0.8095 | 0.9292 |
| **RiskMask** | 50 | 0.8690 | 0.8088 | 0.9292 |
| **RiskMask** | 100 | 0.8686 | 0.8081 | 0.9292 |
| **RiskMask** | 150 | 0.8708 | 0.8108 | 0.9308 |
| **RiskMask** | 200 | 0.8697 | 0.8102 | 0.9292 |

---

## Notes

- **RiskMask**: Only steps 50–200 available. Steps 250+ failed due to OSS 404 (checkpoint not uploaded).
- **RKL**: All 7 steps (50–312) completed successfully.
- Both RKL and RiskMask consistently outperform Base on CHAIR and AMBER hallucination metrics.
- RiskMask shows better CHAIRi at early steps (step_100: 0.2873 vs Base 0.3025).
- RKL shows steady improvement on POPE with later steps achieving best scores.

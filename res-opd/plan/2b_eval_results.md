# 2B Model Evaluation Results

> Generated: 2026-07-09
> Dataset: train5000_test1000 (Instruct Base) / train5000_test1000_original_sr1p0 (others)

---

## 2B Instruct

### CHAIR (COCO Captioning)

| Model | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|
| **Base** | 0.2293 | 0.0878 | 0.7707 | 0.7010 | **0.7342** | 0.0582 |
| **Base (new)** | 0.2274 | 0.0968 | 0.7726 | 0.6983 | 0.7336 | 0.0627 |
| **OPD s50** | 0.2232 | 0.1070 | 0.7768 | 0.6900 | 0.7308 | **0.0520** |
| **OPD s100** | **0.2217** | **0.0811** | **0.7783** | 0.6924 | 0.7328 | 0.0761 |
| **OPD s150** | 0.2287 | 0.0857 | 0.7713 | 0.6769 | 0.7210 | 0.0755 |
| **riskmask s50** | **0.2155** | **0.0548** | **0.7845** | 0.6841 | 0.7309 | **0.0505** |
| **riskmask s100** | 0.2314 | 0.1138 | 0.7686 | 0.6838 | 0.7237 | 0.0730 |
| **riskmask s150** | 0.2233 | 0.0848 | 0.7767 | 0.6910 | 0.7314 | 0.0688 |

### POPE (Object Hallucination)

| Model | adv Acc↑ | adv P↑ | adv R↑ | adv F1↑ | pop Acc↑ | pop P↑ | pop R↑ | pop F1↑ | rand Acc↑ | rand P↑ | rand R↑ | rand F1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Base** | 0.9154 | 0.8712 | 0.9426 | 0.9055 | 0.9451 | 0.9287 | 0.9448 | 0.9367 | 0.9625 | 0.9685 | 0.9435 | 0.9558 |
| **Base (new)** | 0.9156 | 0.8713 | 0.9430 | 0.9057 | 0.9449 | 0.9275 | 0.9457 | 0.9365 | 0.9621 | 0.9672 | 0.9439 | 0.9554 |
| **OPD s50** | **0.9181** | **0.8791** | 0.9386 | 0.9079 | 0.9451 | 0.9321 | 0.9408 | 0.9364 | 0.9625 | 0.9715 | 0.9404 | 0.9557 |
| **OPD s100** | 0.9169 | 0.8744 | 0.9421 | 0.9070 | 0.9454 | 0.9295 | 0.9448 | 0.9371 | 0.9625 | 0.9676 | 0.9444 | 0.9559 |
| **OPD s150** | **0.9179** | **0.8759** | 0.9426 | **0.9080** | 0.9443 | 0.9286 | 0.9430 | 0.9357 | **0.9634** | **0.9698** | 0.9444 | **0.9569** |
| **riskmask s50** | 0.9173 | 0.8777 | 0.9386 | 0.9071 | 0.9452 | 0.9310 | 0.9426 | 0.9368 | 0.9609 | 0.9684 | 0.9399 | 0.9539 |
| **riskmask s100** | 0.9161 | 0.8752 | 0.9390 | 0.9060 | 0.9443 | 0.9297 | 0.9417 | 0.9357 | 0.9625 | 0.9698 | 0.9421 | 0.9558 |
| **riskmask s150** | 0.9161 | 0.8733 | 0.9417 | 0.9062 | 0.9437 | 0.9277 | 0.9426 | 0.9351 | 0.9625 | 0.9698 | 0.9421 | 0.9558 |

### AMBER (Hallucination Benchmark)

#### Generative Tasks

| Model | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|
| **Base** | 5.5 | **63.4** | 37.5 | 2.9 |
| **Base (new)** | 5.7 | **63.7** | 38.0 | **2.6** |
| **OPD s50** | **3.2** | 63.2 | **35.3** | 2.9 |
| **OPD s100** | 8.5 | 62.8 | 37.6 | 2.7 |
| **OPD s150** | 6.6 | 63.0 | 37.1 | 2.8 |
| **riskmask s50** | 4.2 | 63.2 | 36.3 | **2.8** |
| **riskmask s100** | 5.6 | 63.2 | 37.6 | 2.9 |
| **riskmask s150** | **3.6** | 63.1 | **35.6** | **2.7** |

#### Discriminative Tasks

| Model | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|
| **Base** | 83.7 | — | — | — |
| **Base (new)** | 83.7 | 86.9 | 88.9 | 87.9 |
| **OPD s50** | 83.9 | 87.2 | 88.9 | 88.0 |
| **OPD s100** | 83.9 | 86.9 | 89.2 | 88.0 |
| **OPD s150** | 83.9 | 86.8 | 89.4 | 88.1 |
| **riskmask s50** | 83.8 | 87.1 | 88.8 | 87.9 |
| **riskmask s100** | 84.0 | **87.0** | 89.2 | 88.1 |
| **riskmask s150** | 84.0 | 86.9 | 89.4 | 88.1 |

#### Sub-category Accuracy

| Model | Existence↑ | Attribute↑ | State↑ | Number↑ |
|---|---|---|---|---|
| **Base** | 92.0 | 80.7 | 77.2 | 86.1 |
| **Base (new)** | 92.0 | 80.7 | 77.2 | 86.1 |
| **OPD s50** | 92.4 | 80.7 | 77.2 | 86.5 |
| **OPD s100** | 92.6 | 80.7 | 77.1 | 86.6 |
| **OPD s150** | 92.7 | 80.6 | 77.0 | 86.6 |
| **riskmask s50** | 92.2 | 80.7 | 77.2 | 86.4 |
| **riskmask s100** | 92.6 | 80.8 | 77.3 | 86.4 |
| **riskmask s150** | 92.6 | 80.8 | 77.1 | 86.5 |

### MMStar (Vision-Language Understanding)

| Model | Overall↑ | Coarse Perc↑ | Fine-grained↑ | Instance Reas↑ | Logical Reas↑ | Math↑ | Sci&Tech↑ |
|---|---|---|---|---|---|---|---|
| **Base** | 0.5433 | 0.700 | 0.524 | 0.668 | 0.448 | 0.520 | 0.400 |
| **Base (new)** | 0.5433 | 0.700 | 0.524 | 0.668 | 0.448 | 0.520 | 0.400 |
| **OPD s50** | 0.5413 | 0.688 | 0.516 | 0.660 | 0.464 | 0.520 | 0.400 |
| **OPD s100** | **0.5460** | 0.700 | 0.524 | 0.668 | 0.448 | 0.524 | 0.400 |
| **OPD s150** | 0.5453 | — | — | — | — | — | — |
| **riskmask s50** | 0.5447 | 0.700 | 0.524 | 0.668 | 0.456 | 0.504 | 0.416 |
| **riskmask s100** | 0.5407 | — | — | — | — | — | — |
| **riskmask s150** | 0.5440 | — | — | — | — | — | — |

### CVBench (Visual Reasoning)

| Model | Overall↑ | 2D↑ | 3D↑ | Count↑ | Relation↑ | Depth↑ | Distance↑ |
|---|---|---|---|---|---|---|---|
| **Base** | 0.8059 | 0.7427 | 0.8692 | 0.6282 | 0.8815 | 0.9350 | 0.8033 |
| **Base (new)** | 0.8059 | 0.7427 | 0.8692 | — | — | — | — |
| **OPD s50** | 0.8058 | 0.7399 | 0.8717 | 0.6269 | 0.8769 | 0.9383 | 0.8050 |
| **OPD s100** | 0.8048 | 0.7413 | 0.8683 | — | — | — | — |
| **OPD s150** | 0.8014 | 0.7420 | 0.8608 | — | — | — | — |
| **riskmask s50** | 0.8059 | 0.7385 | 0.8733 | — | — | — | — |
| **riskmask s100** | 0.8054 | 0.7434 | 0.8675 | — | — | — | — |
| **riskmask s150** | 0.8035 | 0.7420 | 0.8650 | — | — | — | — |

---

## 2B Thinking

### CHAIR (COCO Captioning)

| Model | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|
| **Base** | 0.2407 | **0.0275** | 0.7593 | **0.6680** | **0.7108** | 0.0141 |
| **OPD s100** | **0.2396** | 0.0289 | **0.7604** | 0.6641 | 0.7090 | **0.0126** |
| **OPD s150** | 0.2478 | 0.0289 | 0.7522 | 0.6589 | 0.7025 | 0.0127 |

### POPE (Object Hallucination)

| Model | adv Acc↑ | adv P↑ | adv R↑ | adv F1↑ | pop Acc↑ | pop P↑ | pop R↑ | pop F1↑ | rand Acc↑ | rand P↑ | rand R↑ | rand F1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Base** | **0.9158** | **0.8906** | **0.9168** | **0.9035** | 0.9318 | 0.9253 | 0.9154 | 0.9204 | 0.9525 | 0.9712 | 0.9168 | 0.9432 |
| **OPD s100** | 0.9114 | 0.8878 | 0.9088 | 0.8982 | 0.9303 | 0.9251 | 0.9119 | 0.9184 | **0.9531** | **0.9735** | 0.9159 | **0.9438** |
| **OPD s150** | 0.9140 | 0.8899 | 0.9132 | 0.9014 | **0.9320** | **0.9269** | **0.9141** | **0.9205** | 0.9525 | 0.9704 | **0.9177** | 0.9433 |

### AMBER (Hallucination Benchmark)

#### Generative Tasks

| Model | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|
| **Base** | 6.1 | **72.3** | 39.8 | 3.0 |
| **OPD s100** | **4.6** | 72.1 | **37.1** | 3.1 |
| **OPD s150** | 7.4 | 72.2 | 39.3 | **2.9** |

#### Discriminative Tasks

| Model | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|
| **Base** | 88.0 | 90.9 | 90.9 | 90.9 |
| **OPD s100** | **88.2** | **91.1** | **91.1** | **91.1** |
| **OPD s150** | 88.1 | 91.0 | 91.1 | 91.0 |

#### Sub-category Accuracy

| Model | Existence↑ | Attribute↑ | State↑ | Number↑ |
|---|---|---|---|---|
| **Base** | 89.6 | **86.2** | **83.2** | 92.8 |
| **OPD s100** | **90.2** | **86.2** | 83.0 | **92.9** |
| **OPD s150** | 90.1 | 86.1 | 82.7 | **93.3** |

### MMStar (Vision-Language Understanding)

| Model | Overall↑ |
|---|---|
| **Base** | 0.5133 |
| **OPD s100** | **0.5153** |
| **OPD s150** | 0.5147 |

### CVBench (Visual Reasoning)

| Model | Overall↑ | 2D↑ | 3D↑ |
|---|---|---|---|
| **Base** | — | — | — |
| **OPD s100** | 0.6521 | 0.6885 | 0.6158 |
| **OPD s150** | **0.6559** | 0.6843 | **0.6275** |

> Note: 2B Thinking Base CVBench result is missing.

---

## Summary

### 2B Instruct Best Models by Metric

| Metric | Best Model | Value | vs Base |
|---|---|---|---|
| CHAIRi↓ | riskmask s50 | 0.2155 | -6.0% |
| CHAIRs↓ | riskmask s50 | 0.0548 | -37.6% |
| RepRate↓ | riskmask s50 | 0.0505 | -13.2% |
| ObjPrec↑ | riskmask s50 | 0.7845 | +1.8% |
| ObjF1↑ | riskmask s150 | 0.7314 | -0.4% |
| AMBER CHAIR↓ | OPD s50 | 3.2 | -41.8% |
| AMBER HAL↓ | OPD s50 | 35.3 | -5.9% |
| AMBER Cover↑ | Base | 63.4 | — |
| AMBER Disc F1↑ | OPD s150 / riskmask s100 / riskmask s150 | 88.1 | N/A (base missing) |
| POPE adv Acc↑ | OPD s50 | 0.9181 | +0.3% |
| MMStar↑ | OPD s100 | 0.5460 | +0.5% |
| CVBench↑ | Base / riskmask s50 | 0.8059 | — |

### 2B Thinking Best Models by Metric

| Metric | Best Model | Value | vs Base |
|---|---|---|---|
| CHAIRi↓ | OPD s100 | 0.2396 | -0.4% |
| CHAIRs↓ | Base | 0.0275 | — |
| ObjF1↑ | Base | 0.7108 | — |
| AMBER CHAIR↓ | OPD s100 | 4.6 | -24.6% |
| AMBER HAL↓ | OPD s100 | 37.1 | -6.8% |
| AMBER Cover↑ | Base | 72.3 | — |
| AMBER Disc F1↑ | OPD s100 | 91.1 | +0.2% |
| POPE adv Acc↑ | Base | 0.9158 | — |
| MMStar↑ | OPD s100 | 0.5153 | +0.4% |
| CVBench↑ | OPD s150 | 0.6559 | N/A (base missing) |

### Key Takeaways

- **Instruct**: riskmask s50 achieves the best CHAIR scores (CHAIRi 0.2155, CHAIRs 0.0548) and lowest RepRate (0.0505). riskmask s150 achieves the best AMBER hallucination reduction (CHAIR 3.6, HAL 35.6). OPD s50 has the best AMBER generative scores (CHAIR 3.2, HAL 35.3) and best POPE adv accuracy. OPD s100 has the best MMStar. Base model's AMBER discriminative F1 is unavailable (original mode); trained models maintain ~87.9–88.1. CVBench is stable (~0.80).
- **Thinking**: OPD s100 is the clear winner with significant AMBER improvements (CHAIR -24.6%, HAL -6.8%) while maintaining competitive discriminative scores and best MMStar. OPD s150 shows degradation on CHAIR/AMBER but best CVBench.
- **POPE** is stable across all models with minimal variation (<0.5%).
- **MMStar** shows very small differences across models (~0.4% max improvement).
- **CVBench**: Instruct models maintain ~0.80 accuracy; Thinking models are significantly lower (~0.65), suggesting thinking mode impacts visual reasoning differently.

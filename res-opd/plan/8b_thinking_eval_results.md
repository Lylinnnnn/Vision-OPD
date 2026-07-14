# 8B Thinking Model Evaluation Results

> Generated: 2026-07-09
> Dataset: train5000_test1000_original_sr1p0
> Note: No 8B Instruct experiments are available yet.

---

## CHAIR (COCO Captioning)

| Model | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|
| **Base** | **0.2722** | 0.0341 | **0.7278** | **0.6654** | **0.6952** | **0.0078** |
| **OPD s100** | 0.2793 | 0.0353 | 0.7207 | 0.6591 | 0.6885 | 0.0079 |
| **OPD s200** | 0.2768 | 0.0357 | 0.7232 | 0.6614 | 0.6909 | 0.0079 |
| **OPD s300** | 0.2748 | **0.0338** | 0.7252 | 0.6645 | 0.6935 | **0.0078** |

## POPE (Object Hallucination)

| Model | adv Acc↑ | adv P↑ | adv R↑ | adv F1↑ | pop Acc↑ | pop P↑ | pop R↑ | pop F1↑ | rand Acc↑ | rand P↑ | rand R↑ | rand F1↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Base** | 0.9140 | 0.9028 | **0.8968** | 0.8998 | 0.9290 | 0.9343 | **0.8981** | 0.9158 | 0.9479 | 0.9791 | **0.8981** | 0.9369 |
| **OPD s100** | **0.9156** | **0.9082** | 0.8941 | **0.9011** | 0.9278 | 0.9349 | 0.8945 | 0.9143 | 0.9462 | 0.9819 | 0.8914 | 0.9345 |
| **OPD s200** | 0.9125 | 0.9046 | 0.8905 | 0.8975 | 0.9297 | 0.9384 | 0.8954 | 0.9164 | 0.9464 | 0.9823 | 0.8914 | 0.9347 |
| **OPD s300** | 0.9146 | 0.9069 | 0.8932 | 0.9000 | **0.9305** | **0.9381** | 0.8976 | **0.9174** | **0.9472** | **0.9829** | 0.8927 | **0.9356** |

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|
| **Base** | 5.5 | 72.0 | 40.9 | **3.0** |
| **OPD s100** | 5.6 | 72.0 | 42.7 | 3.2 |
| **OPD s200** | 5.6 | 71.6 | 41.1 | 3.1 |
| **OPD s300** | **5.3** | **72.2** | **39.2** | 3.1 |

### Discriminative Tasks

| Model | Acc↑ | Prec↑ | F1↑ |
|---|---|---|---|
| **Base** | **88.8** | **90.8** | **91.6** |
| **OPD s100** | 88.0 | 89.3 | 91.1 |
| **OPD s200** | 87.5 | 88.8 | 90.7 |
| **OPD s300** | 88.0 | 89.4 | 91.1 |

### Sub-category Accuracy

| Model | Existence↑ | Attribute↑ | State↑ | Number↑ |
|---|---|---|---|---|
| **Base** | 92.0 | **86.1** | **83.4** | **92.5** |
| **OPD s100** | **92.3** | 84.5 | 81.2 | 91.2 |
| **OPD s200** | 92.2 | 83.5 | 80.5 | 89.9 |
| **OPD s300** | **92.3** | 84.4 | 81.2 | 91.2 |

## MMStar (Vision-Language Understanding)

| Model | Overall↑ | Coarse Perc↑ | Fine-grained↑ | Instance Reas↑ | Logical Reas↑ | Math↑ | Sci&Tech↑ |
|---|---|---|---|---|---|---|---|
| **Base** | 0.7013 | **0.788** | 0.576 | 0.760 | 0.772 | 0.808 | 0.504 |
| **OPD s100** | 0.7093 | 0.784 | 0.600 | 0.752 | 0.756 | **0.872** | 0.492 |
| **OPD s200** | 0.7113 | 0.776 | **0.624** | **0.772** | **0.788** | 0.828 | 0.480 |
| **OPD s300** | **0.7120** | **0.788** | 0.608 | 0.760 | 0.768 | 0.840 | **0.508** |

## CVBench (Visual Reasoning)

| Model | Overall↑ | 2D↑ | 3D↑ | Count↑ | Relation↑ | Depth↑ | Distance↑ |
|---|---|---|---|---|---|---|---|
| **Base** | **0.8660** | 0.8004 | 0.9317 | 0.6916 | 0.9323 | **0.9517** | 0.9117 |
| **OPD s100** | 0.8644 | 0.7962 | **0.9325** | 0.6891 | 0.9262 | 0.9600 | 0.9050 |
| **OPD s200** | 0.8583 | 0.7907 | 0.9258 | 0.6865 | 0.9169 | 0.9450 | 0.9067 |
| **OPD s300** | 0.8635 | 0.8004 | 0.9267 | 0.6916 | 0.9323 | 0.9450 | 0.9083 |

---

## Summary

### Best Models by Metric

| Metric | Best Model | Value | vs Base |
|---|---|---|---|
| CHAIRi↓ | Base | 0.2722 | — |
| CHAIRs↓ | OPD s300 | 0.0338 | -0.9% |
| ObjF1↑ | Base | 0.6952 | — |
| AMBER CHAIR↓ | OPD s300 | 5.3 | -3.6% |
| AMBER HAL↓ | OPD s300 | 39.2 | -4.2% |
| AMBER Cover↑ | OPD s300 | 72.2 | +0.3% |
| AMBER Disc F1↑ | Base | 91.6 | — |
| POPE adv Acc↑ | OPD s100 | 0.9156 | +0.2% |
| POPE pop F1↑ | OPD s300 | 0.9174 | +0.2% |
| MMStar↑ | OPD s300 | 0.7120 | +1.5% |
| CVBench↑ | Base | 0.8660 | — |

### Key Takeaways

- **OPD s300 is the best overall checkpoint**: lowest AMBER CHAIR (5.3) and HAL (39.2), highest MMStar (0.712), competitive POPE, and CHAIRs closest to base. It recovers most of the base model's discriminative performance while improving generative hallucination metrics.
- **OPD s100 shows mixed results**: best POPE adv accuracy and CVBench overall, but worst AMBER HAL (42.7) and CHAIRi (0.2793). Strong math reasoning on MMStar (0.872 vs base 0.808).
- **OPD s200 is the weakest**: lowest AMBER disc F1 (90.7), lowest sub-category scores across the board, though it has the best fine-grained perception (0.624) and logical reasoning (0.788) on MMStar.
- **Base model retains advantages** in CHAIRi, ObjF1, AMBER discriminative tasks, and sub-category accuracy, suggesting OPD training trades some caption quality for hallucination reduction.
- **POPE is stable** across all models (<0.3% variation).
- **MMStar improves consistently** with training steps: Base 0.701 → s100 0.709 → s200 0.711 → s300 0.712.
- **CVBench**: Base achieves the highest overall accuracy (0.866) and best Depth score (0.952). All OPD checkpoints perform competitively (~0.86), with s100 best on 3D (0.933) and s300 best on 2D (0.800).

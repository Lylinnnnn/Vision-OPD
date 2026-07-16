# 2B Instruct Fixed-Teacher Distillation Results

> Dataset: train5000_test1000, sr=1.0. Student: Qwen3VL-2B-Instruct, Teacher: Qwen3VL-4B-Instruct. Compared with 2B Base and 2B original-image RKL (tr=1.0).

## CHAIR (COCO Captioning)

| Model | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|
| **2B Base** | — | 0.2293 | 0.0878 | 0.7707 | 0.7010 | 0.7342 | 0.0582 |
| **2B RKL (orig, tr=1.0)** | 50 | 0.2327 | 0.0818 | 0.7673 | 0.6993 | 0.7317 | 0.0614 |
| **2B RKL (orig, tr=1.0)** | 100 | 0.2212 | 0.0840 | 0.7788 | 0.6993 | 0.7369 | 0.0591 |
| **2B RKL (orig, tr=1.0)** | 150 | 0.2244 | 0.0867 | 0.7756 | 0.6962 | 0.7338 | 0.0561 |
| **2B RKL (orig, tr=1.0)** | 156 | 0.2209 | 0.0899 | 0.7791 | 0.6979 | 0.7363 | 0.0634 |
| **2B fixed4b (tr=1.0)** | 50 | 0.2486 | 0.0545 | 0.7514 | 0.7169 | 0.7337 | 0.0594 |
| **2B fixed4b (tr=1.0)** | 100 | 0.2479 | 0.0788 | 0.7521 | 0.7166 | 0.7339 | 0.0870 |
| **2B fixed4b (tr=1.0)** | 150 | 0.2490 | 0.0831 | 0.7510 | 0.7166 | 0.7334 | 0.0689 |
| **2B fixed4b (tr=1.0)** | 156 | 0.2529 | 0.0607 | 0.7471 | 0.7183 | 0.7324 | 0.0752 |

## POPE (Object Hallucination)

### Overall

| Model | Step | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|
| **2B Base** | — | 0.9410 | 0.9436 | 0.9327 |
| **2B RKL (orig, tr=1.0)** | 50 | 0.9413 | 0.9447 | 0.9331 |
| **2B RKL (orig, tr=1.0)** | 100 | 0.9412 | 0.9442 | 0.9330 |
| **2B RKL (orig, tr=1.0)** | 150 | 0.9408 | 0.9438 | 0.9325 |
| **2B RKL (orig, tr=1.0)** | 156 | 0.9415 | 0.9445 | 0.9333 |
| **2B fixed4b (tr=1.0)** | 50 | 0.9421 | 0.9401 | 0.9336 |
| **2B fixed4b (tr=1.0)** | 100 | 0.9403 | 0.9450 | 0.9321 |
| **2B fixed4b (tr=1.0)** | 150 | 0.9397 | 0.9466 | 0.9315 |
| **2B fixed4b (tr=1.0)** | 156 | 0.9401 | 0.9470 | 0.9320 |

### Per-Split Accuracy

| Model | Step | adv↑ | pop↑ | random↑ |
|---|---|---|---|---|
| **2B Base** | — | 0.9154 | 0.9451 | 0.9625 |
| **2B RKL (orig, tr=1.0)** | 50 | 0.9156 | 0.9460 | 0.9623 |
| **2B RKL (orig, tr=1.0)** | 100 | 0.9158 | 0.9452 | 0.9627 |
| **2B RKL (orig, tr=1.0)** | 150 | 0.9150 | 0.9447 | 0.9629 |
| **2B RKL (orig, tr=1.0)** | 156 | 0.9165 | 0.9456 | 0.9625 |
| **2B fixed4b (tr=1.0)** | 50 | 0.9179 | 0.9481 | 0.9604 |
| **2B fixed4b (tr=1.0)** | 100 | 0.9140 | 0.9458 | 0.9611 |
| **2B fixed4b (tr=1.0)** | 150 | 0.9137 | 0.9454 | 0.9600 |
| **2B fixed4b (tr=1.0)** | 156 | 0.9142 | 0.9452 | 0.9609 |

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|
| **2B Base** | — | 5.5 | 63.4 | 37.5 | 2.9 |
| **2B RKL (orig, tr=1.0)** | 50 | 11.5 | 63.6 | 37.1 | 2.6 |
| **2B RKL (orig, tr=1.0)** | 100 | 6.0 | 63.5 | 36.8 | 2.8 |
| **2B RKL (orig, tr=1.0)** | 150 | 5.8 | 63.7 | 37.4 | 2.6 |
| **2B RKL (orig, tr=1.0)** | 156 | 10.4 | 63.5 | 37.3 | 2.5 |
| **2B fixed4b (tr=1.0)** | 50 | 4.5 | 64.2 | 42.4 | 2.8 |
| **2B fixed4b (tr=1.0)** | 100 | 6.1 | 64.7 | 45.1 | 3.3 |
| **2B fixed4b (tr=1.0)** | 150 | 7.3 | 64.6 | 46.2 | 3.3 |
| **2B fixed4b (tr=1.0)** | 156 | 6.9 | 65.2 | 46.3 | 3.4 |

### Discriminative Tasks

| Model | Step | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| **2B Base** | — | 83.7 | 86.9 | 88.8 | 87.8 |
| **2B RKL (orig, tr=1.0)** | 50 | 83.7 | 86.9 | 88.8 | 87.8 |
| **2B RKL (orig, tr=1.0)** | 100 | 83.7 | 86.9 | 88.8 | 87.8 |
| **2B RKL (orig, tr=1.0)** | 150 | 83.8 | 86.9 | 88.9 | 87.9 |
| **2B RKL (orig, tr=1.0)** | 156 | 83.7 | 86.9 | 88.9 | 87.9 |
| **2B fixed4b (tr=1.0)** | 50 | 84.1 | 87.5 | 88.7 | 88.1 |
| **2B fixed4b (tr=1.0)** | 100 | 84.3 | 87.8 | 88.7 | 88.2 |
| **2B fixed4b (tr=1.0)** | 150 | 84.4 | 87.8 | 88.9 | 88.3 |
| **2B fixed4b (tr=1.0)** | 156 | 84.3 | 87.7 | 88.8 | 88.2 |

## MMStar (Vision-Language Understanding)

| Model | Step | Overall↑ |
|---|---|---|
| **2B Base** | — | 0.5433 |
| **2B RKL (orig, tr=1.0)** | 50 | 0.5447 |
| **2B RKL (orig, tr=1.0)** | 100 | 0.5480 |
| **2B RKL (orig, tr=1.0)** | 150 | 0.5500 |
| **2B RKL (orig, tr=1.0)** | 156 | 0.5467 |
| **2B fixed4b (tr=1.0)** | 50 | 0.5320 |
| **2B fixed4b (tr=1.0)** | 100 | 0.5253 |
| **2B fixed4b (tr=1.0)** | 150 | 0.5120 |
| **2B fixed4b (tr=1.0)** | 156 | 0.5127 |

## CV-Bench (Visual Reasoning)

| Model | Step | Overall↑ |
|---|---|---|
| **2B Base** | — | 0.8059 |
| **2B RKL (orig, tr=1.0)** | 50 | 0.8071 |
| **2B RKL (orig, tr=1.0)** | 100 | 0.8061 |
| **2B RKL (orig, tr=1.0)** | 150 | 0.8075 |
| **2B RKL (orig, tr=1.0)** | 156 | 0.8066 |
| **2B fixed4b (tr=1.0)** | 50 | 0.7999 |
| **2B fixed4b (tr=1.0)** | 100 | 0.8011 |
| **2B fixed4b (tr=1.0)** | 150 | 0.7947 |
| **2B fixed4b (tr=1.0)** | 156 | 0.7952 |

---

# 4B Instruct Fixed-Teacher Distillation Results

> Dataset: train5000_test1000, sr=1.0. Student: Qwen3VL-4B-Instruct, Teacher: Qwen3VL-8B-Instruct. Compared with 4B Base. (4B original-image RKL tr=1.0 to be added.)

## CHAIR (COCO Captioning)

| Model | Step | CHAIRi↓ | CHAIRs↓ | ObjPrec↑ | ObjRecall↑ | ObjF1↑ | RepRate↓ |
|---|---|---|---|---|---|---|---|
| **4B Base** | — | 0.2823 | 0.0584 | 0.7177 | 0.7338 | 0.7257 | 0.0187 |
| **4B fixed8b (tr=1.0)** | 50 | 0.2759 | 0.0771 | 0.7241 | 0.7303 | 0.7272 | 0.0375 |
| **4B fixed8b (tr=1.0)** | 100 | 0.2787 | 0.0659 | 0.7213 | 0.7417 | 0.7314 | 0.0319 |
| **4B fixed8b (tr=1.0)** | 150 | 0.2821 | 0.1064 | 0.7179 | 0.7334 | 0.7256 | 0.0374 |

## POPE (Object Hallucination)

### Overall

| Model | Step | Accuracy↑ | Recall↑ | F1↑ |
|---|---|---|---|---|
| **4B Base** | — | 0.9439 | 0.9283 | 0.9346 |
| **4B fixed8b (tr=1.0)** | 50 | 0.9437 | 0.9281 | 0.9344 |
| **4B fixed8b (tr=1.0)** | 100 | 0.9438 | 0.9279 | 0.9345 |
| **4B fixed8b (tr=1.0)** | 150 | 0.9431 | 0.9258 | 0.9336 |

### Per-Split Accuracy

| Model | Step | adv↑ | pop↑ | random↑ |
|---|---|---|---|---|
| **4B Base** | — | 0.9271 | 0.9429 | 0.9617 |
| **4B fixed8b (tr=1.0)** | 50 | 0.9259 | 0.9429 | 0.9623 |
| **4B fixed8b (tr=1.0)** | 100 | 0.9269 | 0.9426 | 0.9621 |
| **4B fixed8b (tr=1.0)** | 150 | 0.9253 | 0.9428 | 0.9611 |

## AMBER (Hallucination Benchmark)

### Generative Tasks

| Model | Step | CHAIR↓ | Cover↑ | HAL↓ | COG↓ |
|---|---|---|---|---|---|
| **4B Base** | — | 6.1 | 65.0 | 44.0 | 3.2 |
| **4B fixed8b (tr=1.0)** | 50 | 6.0 | 64.9 | 45.3 | 3.4 |
| **4B fixed8b (tr=1.0)** | 100 | 6.8 | 64.9 | 47.2 | 2.9 |
| **4B fixed8b (tr=1.0)** | 150 | 5.9 | 65.8 | 45.1 | 3.5 |

### Discriminative Tasks

| Model | Step | Acc↑ | Prec↑ | Recall↑ | F1↑ |
|---|---|---|---|---|---|
| **4B Base** | — | 86.1 | 86.5 | 93.6 | 89.9 |
| **4B fixed8b (tr=1.0)** | 50 | 86.1 | 86.5 | 93.6 | 89.9 |
| **4B fixed8b (tr=1.0)** | 100 | 86.0 | 86.3 | 93.7 | 89.8 |
| **4B fixed8b (tr=1.0)** | 150 | 86.0 | 86.4 | 93.7 | 89.9 |

## MMStar (Vision-Language Understanding)

| Model | Step | Overall↑ |
|---|---|---|
| **4B Base** | — | 0.6167 |
| **4B fixed8b (tr=1.0)** | 50 | 0.6047 |
| **4B fixed8b (tr=1.0)** | 100 | 0.6067 |
| **4B fixed8b (tr=1.0)** | 150 | 0.6020 |

## CV-Bench (Visual Reasoning)

| Model | Step | Overall↑ |
|---|---|---|
| **4B Base** | — |  |
| **4B fixed8b (tr=1.0)** | 50 | 0.8599 |
| **4B fixed8b (tr=1.0)** | 100 | 0.8592 |
| **4B fixed8b (tr=1.0)** | 150 | 0.8549 |
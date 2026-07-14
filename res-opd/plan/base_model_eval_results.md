## Base Model Evaluation Results (Instruct, Full Dataset)

Evaluation benchmarks: CHAIR, POPE, AMBER
Dataset: train5000_test1000, original degradation mode

### Qwen3VL-8B-Instruct

#### CHAIR
| Metric | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|--------|-------|-------|-------|-------|
| CHAIRi | 0.2804 | 0.2925 | 0.2948 | 0.3025 |
| CHAIRs | 0.0522 | 0.0585 | 0.0690 | 0.0597 |
| ObjPrec | 0.7196 | 0.7075 | 0.7052 | 0.6975 |
| ObjRecall | 0.6814 | 0.7207 | 0.7324 | 0.7393 |
| ObjF1 | 0.7000 | 0.7140 | 0.7185 | 0.7178 |
| RepRate | 0.0099 | 0.0138 | 0.0140 | 0.0156 |

#### POPE
**Overall:**
| Metric | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|--------|-------|-------|-------|-------|
| Accuracy | 0.9234 | 0.9380 | 0.9366 | 0.9382 |
| Recall | 0.8953 | 0.9304 | 0.9334 | 0.9386 |
| F1 | 0.9100 | 0.9286 | 0.9273 | 0.9294 |

**Per-Split Accuracy:**
| Split | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|-------|-------|-------|-------|-------|
| pope_adv | 0.8999 | 0.9146 | 0.9125 | 0.9125 |
| pope_pop | 0.9253 | 0.9397 | 0.9366 | 0.9376 |
| pope_random | 0.9451 | 0.9598 | 0.9608 | 0.9644 |

#### AMBER - Generative Tasks
| Task | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|------|-------|-------|-------|-------|
| CHAIR | 5.6 | 5.8 | 5.9 | 6.0 |
| COVER | 61.9 | 64.4 | 64.8 | 64.7 |
| HAL | 38.3 | 44.5 | 44.6 | 46.8 |
| COG | 2.4 | 2.5 | 3.0 | 2.5 |

#### AMBER - Discriminative Tasks
| Category | Metric | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|----------|--------|-------|-------|-------|-------|
| Exsitence | Accuracy | 93.5 | 93.6 | 93.4 | 93.3 |
| Exsitence | Precision | 100.0 | 100.0 | 100.0 | 100.0 |
| Exsitence | Recall | 93.5 | 93.6 | 93.4 | 93.3 |
| Exsitence | F1 | 96.6 | 96.6 | 96.5 | 96.5 |
| Attribute | Accuracy | 81.0 | 81.3 | 81.4 | 81.4 |
| Attribute | Precision | 76.7 | 77.0 | 76.9 | 77.0 |
| Attribute | Recall | 89.2 | 89.3 | 89.8 | 89.6 |
| Attribute | F1 | 82.5 | 82.7 | 82.8 | 82.8 |
| State | Accuracy | 78.1 | 78.1 | 78.4 | 78.4 |
| State | Precision | 74.4 | 74.4 | 74.5 | 74.6 |
| State | Recall | 85.7 | 85.8 | 86.5 | 86.3 |
| State | F1 | 79.6 | 79.7 | 80.0 | 80.0 |
| Number | Accuracy | 85.7 | 86.1 | 86.0 | 85.9 |
| Number | Precision | 79.8 | 80.4 | 80.2 | 80.2 |
| Number | Recall | 95.5 | 95.3 | 95.6 | 95.2 |
| Number | F1 | 86.9 | 87.2 | 87.2 | 87.1 |
| Action | Accuracy | 86.5 | 87.9 | 87.1 | 87.6 |
| Action | Precision | 81.9 | 83.5 | 82.5 | 82.8 |
| Action | Recall | 93.7 | 94.4 | 94.2 | 94.9 |
| Action | F1 | 87.4 | 88.6 | 88.0 | 88.4 |
| Relation | Accuracy | 79.9 | 80.2 | 80.5 | 80.4 |
| Relation | Precision | 68.2 | 68.4 | 68.8 | 68.6 |
| Relation | Recall | 96.1 | 96.7 | 96.8 | 97.2 |
| Relation | F1 | 79.8 | 80.1 | 80.4 | 80.4 |
| **Overall** | Accuracy | **85.2** | **85.4** | **85.4** | **85.4** |
| **Overall** | Precision | **86.6** | **86.8** | **86.7** | **86.7** |
| **Overall** | Recall | **92.0** | **92.0** | **92.2** | **92.1** |
| **Overall** | F1 | **89.2** | **89.3** | **89.4** | **89.3** |

---

### Qwen3VL-2B-Instruct

#### CHAIR
| Metric | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|--------|-------|-------|-------|-------|
| CHAIRi | 0.2150 | 0.2320 | 0.2262 | 0.2274 |
| CHAIRs | 0.0697 | 0.0730 | 0.0708 | 0.0968 |
| ObjPrec | 0.7850 | 0.7680 | 0.7738 | 0.7726 |
| ObjRecall | 0.6269 | 0.6793 | 0.6890 | 0.6983 |
| ObjF1 | 0.6971 | 0.7210 | 0.7289 | 0.7336 |
| RepRate | 0.0433 | 0.0573 | 0.0646 | 0.0627 |

#### POPE
**Overall:**
| Metric | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|--------|-------|-------|-------|-------|
| Accuracy | 0.9281 | 0.9383 | 0.9399 | 0.9408 |
| Recall | 0.8951 | 0.9282 | 0.9372 | 0.9442 |
| F1 | 0.9149 | 0.9286 | 0.9311 | 0.9326 |

**Per-Split Accuracy:**
| Split | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|-------|-------|-------|-------|-------|
| pope_adv | 0.9083 | 0.9159 | 0.9148 | 0.9156 |
| pope_pop | 0.9284 | 0.9403 | 0.9424 | 0.9449 |
| pope_random | 0.9475 | 0.9586 | 0.9627 | 0.9621 |

#### AMBER - Generative Tasks
| Task | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|------|-------|-------|-------|-------|
| CHAIR | 5.3 | 6.5 | 8.5 | 5.7 |
| COVER | 60.9 | 62.5 | 63.2 | 63.7 |
| HAL | 33.6 | 36.3 | 40.2 | 38.0 |
| COG | 2.5 | 2.5 | 2.9 | 2.6 |

#### AMBER - Discriminative Tasks
| Category | Metric | sr=0.25 | sr=0.5 | sr=0.75 | sr=1.0 |
|----------|--------|-------|-------|-------|-------|
| Exsitence | Accuracy | 93.3 | 93.0 | 92.8 | 92.0 |
| Exsitence | Precision | 100.0 | 100.0 | 100.0 | 100.0 |
| Exsitence | Recall | 93.3 | 93.0 | 92.8 | 92.0 |
| Exsitence | F1 | 96.5 | 96.3 | 96.2 | 95.8 |
| Attribute | Accuracy | 79.9 | 80.5 | 80.6 | 80.7 |
| Attribute | Precision | 77.8 | 78.7 | 78.6 | 78.9 |
| Attribute | Recall | 83.6 | 83.8 | 84.1 | 83.8 |
| Attribute | F1 | 80.6 | 81.2 | 81.3 | 81.3 |
| State | Accuracy | 76.6 | 77.2 | 77.0 | 77.2 |
| State | Precision | 75.3 | 75.9 | 75.6 | 75.9 |
| State | Recall | 79.1 | 79.7 | 79.8 | 79.7 |
| State | F1 | 77.1 | 77.7 | 77.6 | 77.7 |
| Number | Accuracy | 85.1 | 85.9 | 86.3 | 86.1 |
| Number | Precision | 81.4 | 82.8 | 83.2 | 83.6 |
| Number | Recall | 91.0 | 90.4 | 91.0 | 90.0 |
| Number | F1 | 85.9 | 86.4 | 86.9 | 86.7 |
| Action | Accuracy | 86.1 | 86.6 | 87.4 | 87.2 |
| Action | Precision | 82.8 | 83.7 | 84.1 | 83.9 |
| Action | Recall | 91.2 | 90.9 | 92.2 | 92.2 |
| Action | F1 | 86.8 | 87.1 | 88.0 | 87.8 |
| Relation | Accuracy | 73.4 | 73.0 | 72.6 | 72.9 |
| Relation | Precision | 61.9 | 61.4 | 61.1 | 61.2 |
| Relation | Recall | 93.0 | 93.8 | 93.3 | 94.0 |
| Relation | F1 | 74.3 | 74.2 | 73.8 | 74.1 |
| **Overall** | Accuracy | **83.8** | **84.0** | **83.9** | **83.7** |
| **Overall** | Precision | **86.6** | **86.9** | **86.8** | **86.9** |
| **Overall** | Recall | **89.3** | **89.3** | **89.3** | **88.9** |
| **Overall** | F1 | **87.9** | **88.1** | **88.0** | **87.9** |

---

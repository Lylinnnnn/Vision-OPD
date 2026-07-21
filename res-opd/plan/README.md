# Res-OPD 实验结果索引

> 本目录包含 Vision-OPD (Res-OPD) 项目的所有实验评估结果文档。
> 数据集统一为 `train5000_test1000`，降质模式为 `original`。
> 评估指标：CHAIR（幻觉率↓）、POPE（对象幻觉↑）、AMBER（幻觉基准）、MMStar（视觉理解↑）、CV-Bench（视觉推理↑）

---

## 📋 文件总览

| 文件 | 涵盖模型 | 核心变量 | 研究问题 |
|------|---------|---------|---------|
| [base_model_eval_results.md](base_model_eval_results.md) | 8B / 2B | sr (0.25/0.5/0.75/1.0) | Base 模型在不同降质程度下的性能基线 |
| [instruct_tr025_eval_results.md](instruct_tr025_eval_results.md) | 2B / 4B / 8B | tr=0.25 | Teacher Ratio=0.25 时 RKL/RiskMask 完整评估 |
| [instruct_tr05_eval_results.md](instruct_tr05_eval_results.md) | 2B / 4B / 8B | tr=0.5 | Teacher Ratio=0.5 时 RKL/RiskMask 完整评估 |
| [instruct_tr075_eval_results.md](instruct_tr075_eval_results.md) | 2B / 4B / 8B | tr=0.75 | Teacher Ratio=0.75 时 RKL/RiskMask 完整评估 |
| [instruct_masking_ablation_results.md](instruct_masking_ablation_results.md) | 2B / 4B / 8B | masking 策略 × tr × 模型尺寸 | 多种 masking 策略在不同 tr 和模型尺寸下的全面消融 |
| [instruct_sft_ablation_results.md](instruct_sft_ablation_results.md) | 2B / 4B / 8B | SFT vs RKL vs RiskMask × tr × 模型尺寸 | SFT 与蒸馏方法在不同 tr 和模型尺寸下的全面对比 |
| [2b_instruct_tr10_masking_results.md](2b_instruct_tr10_masking_results.md) | 2B | masking 策略 (tr=1.0) | tr=1.0 下 RiskMask-NLL vs 纯 RKL 的对比 |
| [2b_instruct_tr10_sr_ablation_results.md](2b_instruct_tr10_sr_ablation_results.md) | 2B | sr (0.25/0.5/0.75, tr=1.0) | tr=1.0 下不同降质程度对 RKL 效果的影响 |
| [2b_sft_randommask_tr075_results.md](2b_sft_randommask_tr075_results.md) | 2B | SFT + RandomMask (tr=0.75) | 2B SFT 结合 RandomMask 的独立评估结果 |
| [8b_riskmask_top_p_ablation_results.md](8b_riskmask_top_p_ablation_results.md) | 8B | top-p (0.10/0.20/0.30/0.40, tr=0.75) | 8B RiskMask 不同 top-p 阈值的敏感性分析 |
| [fixed_teacher_distillation_results.md](fixed_teacher_distillation_results.md) | 2B student ← 4B teacher | fixed teacher (tr=1.0) | 固定 4B teacher 蒸馏到 2B student 的效果 |
| [fixed_teacher_distillation_tr075_results.md](fixed_teacher_distillation_tr075_results.md) | 2B←4B / 4B←8B | fixed teacher (tr=0.75) | 跨尺寸固定 teacher 蒸馏 (tr=0.75) |

---

## 🔬 实验分类详解

### 一、Teacher Ratio (tr) 扫描实验

研究 **teacher 输出在训练损失中的权重** 对蒸馏效果的影响。tr 越高，student 越依赖 teacher 的分布；tr 越低，越依赖原始 SFT 目标。

- **[instruct_tr025_eval_results.md](instruct_tr025_eval_results.md)** — tr=0.25，teacher 影响较弱，接近纯 SFT
- **[instruct_tr05_eval_results.md](instruct_tr05_eval_results.md)** — tr=0.5，teacher 和 student 各占一半
- **[instruct_tr075_eval_results.md](instruct_tr075_eval_results.md)** — tr=0.75，teacher 主导，当前主线配置

每个文件包含 **2B / 4B / 8B Instruct** 三个模型尺寸的 RKL 和 RiskMask 完整评估（CHAIR、POPE、AMBER、MMStar、CV-Bench），以及多 step checkpoint 的性能变化趋势。

### 二、Masking 策略消融实验

研究 **不同的 token-level masking/weighting 策略** 如何影响蒸馏质量。核心思路是在 RKL 损失中对"高风险"token 进行选择性加权或遮蔽，以减少错误知识的传递。

#### [instruct_masking_ablation_results.md](instruct_masking_ablation_results.md)

该文件包含 **5 组实验**，覆盖不同模型尺寸和 tr 组合：

| 章节 | 模型 | tr | Masking 策略 |
|------|------|-----|-------------|
| 2B Masking Ablation (tr=0.5) | 2B-Instruct | 0.5 | RKL, RiskMask, BlankTeacher, EntropyMask, RandomMask, RiskEntropy（6种） |
| 8B Masking Ablation (tr=0.5) | 8B-Instruct | 0.5 | RKL, RiskMask, RandomMask, EntropyMask（4种） |
| 8B Masking Ablation (tr=0.75) | 8B-Instruct | 0.75 | RKL, RiskMask, RandomMask, EntropyMask（4种） |
| 4B Masking Ablation (tr=0.75) | 4B-Instruct | 0.75 | RKL, RiskMask, RandomMask, EntropyMask（4种） |
| 2B Masking Ablation (tr=0.75, b=32) | 2B-Instruct | 0.75 | RKL, RiskMask, RandomMask, EntropyMask（4种，batch=32） |

每组实验均包含 CHAIR、POPE（Overall + Per-Split）、AMBER（Generative + Discriminative）、MMStar、CV-Bench 完整指标。

**Masking 策略说明：**
- **RKL**：标准反向 KL 散度，无 masking（基线）
- **RiskMask (NLL)**：基于 NLL 的 risk-based masking，遮蔽高 loss token
- **EntropyMask**：基于 entropy 的 masking
- **RandomMask**：随机 masking（对照实验）
- **RiskEntropy**：结合 risk 和 entropy 的混合策略（仅 2B tr=0.5）
- **BlankTeacher**：tr=0.0 空白 teacher，纯 SFT 对照（仅 2B tr=0.5）

#### [2b_instruct_tr10_masking_results.md](2b_instruct_tr10_masking_results.md)

tr=1.0 下 **RiskMask-NLL vs 纯 RKL** 的直接对比，验证 masking 在高 tr 下的效果。包含 CHAIR、POPE、AMBER、MMStar、CV-Bench 完整指标。

### 三、Student Ratio (sr) / 降质程度消融

研究 **输入图像的降质程度** 对蒸馏效果的影响。sr=1.0 为原图，sr 越低图像质量越差，模拟更极端的退化场景。

- **[base_model_eval_results.md](base_model_eval_results.md)** — **8B-Instruct 和 2B-Instruct** Base 模型在 sr=0.25/0.5/0.75/1.0 下的性能基线（CHAIR、POPE、AMBER），作为所有实验的参照
- **[2b_instruct_tr10_sr_ablation_results.md](2b_instruct_tr10_sr_ablation_results.md)** — 2B tr=1.0 RKL 在 sr=0.25/0.5/0.75 下的消融，观察降质程度如何影响蒸馏收益。⚠️ 部分数据不完整（sr=0.25 AMBER discriminative 为 0.0；sr=0.5/0.75 step 156 的 AMBER/MMStar/CV-Bench 缺失）

### 四、SFT vs 蒸馏方法对比

研究 **传统 SFT（用降质图 caption 微调）** 与 **RKL/RiskMask 蒸馏** 的效果差异。SFT 直接在降质图的 caption 上训练，而蒸馏方法通过 teacher-student 分布对齐来学习。

#### [instruct_sft_ablation_results.md](instruct_sft_ablation_results.md)

该文件包含 **6 个部分**，覆盖不同模型尺寸和 tr 组合：

| 章节 | 模型 | tr | 对比方法 |
|------|------|-----|---------|
| 2B SFT (tr=0.75) | 2B-Instruct | 0.75 | Base vs RKL vs RiskMask vs SFT（step 156） |
| 2B SFT (tr=0.5) | 2B-Instruct | 0.5 | Base vs RKL vs RiskMask vs SFT（多 step） |
| 4B SFT (tr=0.5) | 4B-Instruct | 0.5 | Base vs RKL vs RiskMask vs SFT（多 step） |
| 4B SFT (tr=0.75) | 4B-Instruct | 0.75 | Base vs RKL vs RiskMask vs SFT（steps 50/100/150/156） |
| Cross-Experiment Summary | 2B/4B 汇总 | 0.5+0.75 | 跨 tr 的 CHAIR/POPE 对比 + Key Findings |
| 8B SFT (tr=0.75) | 8B-Instruct | 0.75 | Base vs RKL vs RiskMask vs SFT + Cross-Model Summary |

每个部分均包含 CHAIR、POPE（Overall + Per-Split）、AMBER（Generative + Discriminative + Sub-category）、MMStar、CV-Bench 完整指标及 Analysis。

#### [2b_sft_randommask_tr075_results.md](2b_sft_randommask_tr075_results.md)

2B SFT 结合 RandomMask 的独立评估结果（tr=0.75），探索 SFT + masking 的组合效果。包含完整的 CHAIR、POPE、AMBER、MMStar、CV-Bench 指标及 Summary & Key Findings。

### 五、RiskMask Top-P 敏感性分析

研究 **RiskMask 中 top-p 阈值** 对 8B 模型蒸馏效果的影响。top-p 控制被 mask 的 token 比例，p 越小 mask 越多（更保守），p 越大 mask 越少（更激进）。

- **[8b_riskmask_top_p_ablation_results.md](8b_riskmask_top_p_ablation_results.md)** — 8B tr=0.75 下 **p=0.10 / 0.20 / 0.30 / 0.40** 的完整对比，加上 RKL 基线。包含 CHAIR、POPE（Per-Split + Overall）、AMBER（Generative + Discriminative）、MMStar、CV-Bench 所有指标的多 step 数据，以及 Top-P Sensitivity Analysis 和 Recommendation。

### 六、Fixed-Teacher 跨尺寸蒸馏

研究 **使用更大尺寸的固定 teacher 蒸馏到更小 student** 的效果，与同尺寸 self-distillation 进行对比。

- **[fixed_teacher_distillation_results.md](fixed_teacher_distillation_results.md)** — tr=1.0 下 **4B teacher → 2B student** 的蒸馏效果，对比 2B Base 和 2B self-RKL (tr=1.0)。包含 CHAIR、POPE、AMBER、MMStar、CV-Bench 完整指标。
- **[fixed_teacher_distillation_tr075_results.md](fixed_teacher_distillation_tr075_results.md)** — tr=0.75 下的跨尺寸蒸馏，包含 **两组实验**：
  - **2B student ← 4B teacher** (tr=0.75)：对比 2B Base 和 2B self-RKL (tr=0.75)
  - **4B student ← 8B teacher** (tr=0.75)：对比 4B Base 和 4B self-RKL (tr=0.75)

---

## 📁 归档文件

`old/` 目录包含已整合到上述文档中的早期实验结果，不再需要单独查阅：
- `2b_eval_results.md` — 早期 2B OPD/RiskMask 评估
- `8b_instruct_eval_results.md` — 早期 8B Instruct 评估
- `8b_thinking_eval_results.md` — 8B Thinking 模型早期评估
- `selective_weighted_rkl.md` — Selective Weighted RKL 方案设计文档

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| **tr (Teacher Ratio)** | Teacher 分布在训练损失中的权重，范围 [0, 1] |
| **sr (Student Ratio)** | 输入图像的降质程度，1.0=原图，越低越模糊 |
| **RKL** | Reverse KL Divergence，标准蒸馏损失 |
| **RiskMask** | 基于 NLL 的 risk-based token masking，遮蔽高 loss token 以减少错误知识传递 |
| **EntropyMask** | 基于 token entropy 的 masking |
| **RandomMask** | 随机选择 token 进行 masking（对照实验） |
| **RiskEntropy** | 结合 risk score 和 entropy 的混合 masking 策略 |
| **BlankTeacher** | tr=0.0，不使用 teacher 信号，等价于纯 SFT |
| **top-p** | RiskMask 中被 mask 的 token 比例阈值 |
| **SFT** | Supervised Fine-Tuning，直接用降质图 caption 微调 |
| **Fixed Teacher** | 使用更大尺寸的预训练模型作为固定 teacher（不更新） |
| **CHAIRi/s** | Caption Hallucination Assessment，instance/sentence 级别幻觉率（↓越好） |
| **POPE** | Polling-based Object-level Pretraining Evaluation，对象幻觉检测（↑越好） |
| **AMBER** | Hallucination Benchmark，综合幻觉评估（Generative + Discriminative） |
| **MMStar** | Vision-Language Understanding 基准（↑越好） |
| **CV-Bench** | Visual Reasoning 基准（↑越好） |

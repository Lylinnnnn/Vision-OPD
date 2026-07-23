# Res-OPD 实验结果索引

> 本目录包含 Vision-OPD (Res-OPD) 项目的所有实验评估结果文档。
> 数据集统一为 `train5000_test1000`，降质模式为 `original`。
> 评估指标：CHAIR（幻觉率↓）、POPE（对象幻觉↑）、AMBER（幻觉基准）、MMStar（视觉理解↑）、CV-Bench（视觉推理↑）
> OPD = **On-Policy Distillation**；本文实验中的 `tr` / `sr` 均表示图像分辨率比例，不表示 loss 权重。

---

## 📋 文件总览

| 文件 | 涵盖模型 | 核心变量 | 研究问题 |
|------|---------|---------|---------|
| [base_model_eval_results.md](base_model_eval_results.md) | 8B / 2B | sr (0.25/0.5/0.75/1.0) | Base 模型在不同降质程度下的性能基线 |
| [instruct_tr025_eval_results.md](instruct_tr025_eval_results.md) | 2B / 4B / 8B | teacher image ratio=0.25 | Teacher 看 25% 降质图时 RKL/RiskMask 完整评估 |
| [instruct_tr05_eval_results.md](instruct_tr05_eval_results.md) | 2B / 4B / 8B | teacher image ratio=0.5 | Teacher 看 50% 降质图时 RKL/RiskMask 完整评估 |
| [instruct_tr075_eval_results.md](instruct_tr075_eval_results.md) | 2B / 4B / 8B | teacher image ratio=0.75 | Teacher 看 75% 降质图时 RKL/RiskMask 完整评估 |
| [instruct_tr10_eval_results.md](instruct_tr10_eval_results.md) | 8B | teacher image ratio=1.0 | Teacher 和 student 都看原图的 no-resolution-gap control |
| [instruct_masking_ablation_results.md](instruct_masking_ablation_results.md) | 2B / 4B / 8B | masking 策略 × tr × 模型尺寸 | 多种 masking 策略在不同 tr 和模型尺寸下的全面消融 |
| [instruct_sft_ablation_results.md](instruct_sft_ablation_results.md) | 2B / 4B / 8B | SFT vs RKL vs RiskMask × tr × 模型尺寸 | SFT 与蒸馏方法在不同 tr 和模型尺寸下的全面对比 |
| [2b_instruct_tr10_masking_results.md](2b_instruct_tr10_masking_results.md) | 2B | masking 策略 (tr=1.0) | tr=1.0 下 RiskMask-NLL vs 纯 RKL 的对比 |
| [2b_instruct_tr10_sr_ablation_results.md](2b_instruct_tr10_sr_ablation_results.md) | 2B | sr (0.25/0.5/0.75, tr=1.0) | tr=1.0 下不同降质程度对 RKL 效果的影响 |
| [8b_riskmask_top_p_ablation_results.md](8b_riskmask_top_p_ablation_results.md) | 8B | top-p (0.10/0.20/0.30/0.40, tr=0.75) | 8B RiskMask 不同 top-p 阈值的敏感性分析 |
| [base_dual_view_probe_results.md](base_dual_view_probe_results.md) | 8B Base probe | original view vs low-res 0.75 view | RiskMask 设计动机主证据：训练前 cross-resolution disagreement / uncertainty 富集幻觉 object mentions |
| [probe_analysis_writing_notes_zh.md](probe_analysis_writing_notes_zh.md) | 8B / 2B probes | RiskMask enrichment × object decomposition | Probe analysis 结果、主文表格口径与写作注意事项 |
| [model_delta_logprob_probe_writing_notes_zh.md](model_delta_logprob_probe_writing_notes_zh.md) | 8B model-delta probes | base-self / Frozen RKL / RiskMask | 训练后分布漂移、top-k overlap、reshape/suppress 机制分析与 `analysis.tex` 写作建议 |
| [model_delta_surgery_probe_writing_notes_zh.md](model_delta_surgery_probe_writing_notes_zh.md) | 8B model-delta probes | object fate × RKL/NLL/entropy/teacher stance | 更细的 token/mention 象限机制分析：RiskMask 如何局部修改 high-risk token 并保留 correct mentions |
| [analysis_section_plan_zh.md](analysis_section_plan_zh.md) | paper writing | Analysis section | `analysis.tex` 中文写作规划、表格放置和叙事口径 |
| [experiments_section_outline_zh.md](experiments_section_outline_zh.md) | paper writing | Experiments section | `experiments.tex` 中文写作大纲、主表/附录表组织建议 |
| [fixed_teacher_distillation_results.md](fixed_teacher_distillation_results.md) | 2B student ← 4B teacher | fixed teacher (tr=1.0) | 固定 4B teacher 蒸馏到 2B student 的效果 |
| [fixed_teacher_distillation_tr075_results.md](fixed_teacher_distillation_tr075_results.md) | 2B←4B / 4B←8B | fixed teacher (tr=0.75) | 跨尺寸固定 teacher 蒸馏 (tr=0.75) |

---

## 🔬 实验分类详解

### 一、Teacher Ratio (tr) / Teacher 视觉分辨率扫描实验

研究 **teacher 所见图像分辨率/压缩率** 对蒸馏效果的影响。所有实验默认 student 仍看原图（`sr=1.0`），teacher 看同一张图的原图比例降质版本：`tr=1.0` 表示 teacher 看原图，`tr=0.75/0.5/0.25` 表示 teacher 看 75%/50%/25% 降质图，`tr=0.0` 表示灰度/空白 teacher。蒸馏强度主要由 `ALPHA` 和 token mask/weight 机制控制，而不是由 `tr` 控制。

- **[instruct_tr025_eval_results.md](instruct_tr025_eval_results.md)** — tr=0.25，teacher 视觉证据很弱，偏保守，容易带来 coverage trade-off
- **[instruct_tr05_eval_results.md](instruct_tr05_eval_results.md)** — tr=0.5，中等降质 teacher，用于观察压缩强度变化
- **[instruct_tr075_eval_results.md](instruct_tr075_eval_results.md)** — tr=0.75，teacher 仍保留较多视觉信息，是当前主线配置之一
- **[instruct_tr10_eval_results.md](instruct_tr10_eval_results.md)** — tr=1.0，teacher 和 student 都看原图，用作 no-resolution-gap / same-image consistency regularization control；当前包含 8B RKL 和 RiskMask p30 全 step 结果

其中 `tr=0.25/0.5/0.75` 文件包含 **2B / 4B / 8B Instruct** 三个模型尺寸的 RKL 和 RiskMask 完整评估（CHAIR、POPE、AMBER、MMStar、CV-Bench），以及多 step checkpoint 的性能变化趋势；`tr=1.0` 目前作为 8B no-resolution-gap control 单独记录。

### 二、Masking 策略消融实验

研究 **不同的 token-level masking/weighting 策略** 如何影响蒸馏质量。核心思路是在 RKL 损失中只保留或强化更值得蒸馏的 token，避免把低分辨率 teacher 的保守性粗暴扩散到所有 token。

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
- **RiskMask (NLL)**：按 `rank(RKL) × rank(student NLL)` 选择 top-p 高风险 token，仅这些 token 保留 RKL，其余 token 的 RKL 权重置 0
- **EntropyMask**：选择 student entropy 最高的 top-p token 保留 RKL
- **RandomMask**：随机选择相同比例 token 保留 RKL（对照实验）
- **RiskEntropy**：按 `rank(RKL) × rank(student entropy)` 选择 top-p token（当前主要用于机制探索）
- **BlankTeacher**：`tr=0.0`，teacher 输入灰度/空白图，用于验证“完全无视觉信息的保守 teacher”是否足够，而不是纯 SFT

#### [2b_instruct_tr10_masking_results.md](2b_instruct_tr10_masking_results.md)

tr=1.0 下 **RiskMask-NLL vs 纯 RKL** 的直接对比，验证 masking 在高 tr 下的效果。包含 CHAIR、POPE、AMBER、MMStar、CV-Bench 完整指标。

### 三、Student Ratio (sr) / 降质程度消融

研究 **输入图像的降质程度** 对蒸馏效果的影响。sr=1.0 为原图，sr 越低图像质量越差，模拟更极端的退化场景。

- **[base_model_eval_results.md](base_model_eval_results.md)** — **8B-Instruct 和 2B-Instruct** Base 模型在 sr=0.25/0.5/0.75/1.0 下的性能基线（CHAIR、POPE、AMBER），作为所有实验的参照
- **[2b_instruct_tr10_sr_ablation_results.md](2b_instruct_tr10_sr_ablation_results.md)** — 2B tr=1.0 RKL 在 sr=0.25/0.5/0.75 下的消融，观察降质程度如何影响蒸馏收益。⚠️ 部分数据不完整（sr=0.25 AMBER discriminative 为 0.0；sr=0.5/0.75 step 156 的 AMBER/MMStar/CV-Bench 缺失）

### 四、SFT vs 蒸馏方法对比

研究 **低分辨率采样 SFT** 与 **RKL/RiskMask 蒸馏** 的效果差异。SFT pipeline 先让 base model 在低分辨率图像上生成 caption，再把这些 caption 作为 hard target，在原图输入下做 SFT；蒸馏方法则在 on-policy 轨迹上通过 teacher-student 分布对齐来学习。

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

### 五、RiskMask Top-P 敏感性分析

研究 **RiskMask 中 top-p 阈值** 对 8B 模型蒸馏效果的影响。当前代码中 top-p 表示“保留 RKL 的高风险 token 比例”：p 越小，实际参与 RKL 的 token 越少，mask 掉的 token 越多；p 越大，RKL 覆盖范围越宽。

- **[8b_riskmask_top_p_ablation_results.md](8b_riskmask_top_p_ablation_results.md)** — 8B tr=0.75 下 **p=0.10 / 0.20 / 0.30 / 0.40** 的完整对比，加上 RKL 基线。包含 CHAIR、POPE（Per-Split + Overall）、AMBER（Generative + Discriminative）、MMStar、CV-Bench 所有指标的多 step 数据，以及 Top-P Sensitivity Analysis 和 Recommendation。

### 六、Probe Analysis / 机制分析写作

研究 **为什么 RiskMask 是合理的 OPD training gate**，而不只是随机少蒸馏 token 或单独的不确定性排序。

- **[base_dual_view_probe_results.md](base_dual_view_probe_results.md)** — 8B Base dual-view 设计动机 probe。Caption 来自 8B Base 原图输出；student/teacher 都是同一个 8B Base，分别输入原图和 low-res 0.75 图。结果显示 hallucinated object mentions 相比 correct object mentions 有更高 RKL、NLL 和 entropy；RiskMask-NLL top-30% 相比 random top-30% 明显富集幻觉 object mentions。
- **[probe_analysis_writing_notes_zh.md](probe_analysis_writing_notes_zh.md)** — Probe 结果的主文口径总控。重点说明：不要把 RiskMask 写成“最强离线 detector”；应写成 OPD-aligned high-risk gate。Entropy-only 在离线 enrichment 上也强，完整 selector ranking 放 appendix。
- **[model_delta_logprob_probe_writing_notes_zh.md](model_delta_logprob_probe_writing_notes_zh.md)** — 三组 8B model-delta logprob probe 的专门写作说明：base-self sanity check、Frozen RKL vs RiskMask 的 selected/unselected distribution shift、object-token 与 mention-fate 解释。重点用于在 `analysis.tex` 中新增“RiskMask 更局部、更保守地 reshape high-risk tail”的机制段。
- **[model_delta_surgery_probe_writing_notes_zh.md](model_delta_surgery_probe_writing_notes_zh.md)** — 对 model-delta trace 的二次机制拆解。重点修正：Frozen RKL 没有训练时 selected token；只能把 RiskMask score 作为事后 offline risk tail 分层。该文件建议用 object role / mention fate × RKL / NLL / entropy / teacher stance 的多象限统计和 examples 来解释 RiskMask 的 targeted distribution surgery。
- **[analysis_section_plan_zh.md](analysis_section_plan_zh.md)** — `analysis.tex` 写作结构。建议主文放三类证据：8B Base dual-view、RiskMask vs random enrichment、final object decomposition。
- **[experiments_section_outline_zh.md](experiments_section_outline_zh.md)** — `experiments.tex` 写作大纲。说明主表、消融表、附录表如何组织，以及哪些数据应放主文、哪些数据应放附录。

### 七、Fixed-Teacher 跨尺寸蒸馏

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
- `2b_sft_randommask_tr075_results.md` — 2B SFT + RandomMask 的旧实验记录，不作为当前论文主证据

---

## 🔑 关键术语

| 术语 | 含义 |
|------|------|
| **OPD** | On-Policy Distillation；在模型当前生成轨迹上进行分布级蒸馏 |
| **tr (Teacher Ratio)** | Teacher 输入图像的 original-ratio 降质比例，`1.0=原图`，`0.75/0.5/0.25=按原图比例下采样再上采样`，`0=灰度/空白图` |
| **sr (Student Ratio)** | 输入图像的降质程度，1.0=原图，越低越模糊 |
| **RKL** | Reverse KL Divergence，标准蒸馏损失 |
| **RiskMask** | 基于 RKL 分歧与 student NLL 的 risk-only token mask：只在 top-p 高风险 token 上保留 RKL |
| **EntropyMask** | 基于 token entropy 的 mask-only 对照：只在 entropy 最高的 top-p token 上保留 RKL |
| **RandomMask** | 随机选择相同比例 token 保留 RKL（对照实验） |
| **RiskEntropy** | 结合 RKL rank 和 entropy rank 的混合 masking 策略 |
| **BlankTeacher** | `tr=0.0`，teacher 看灰度/空白图，用于测试无视觉 teacher 的下界 |
| **top-p** | mask-only 策略中保留 RKL 的 token 比例阈值 |
| **SFT** | Supervised Fine-Tuning；这里特指“低分辨率生成 caption，原图输入上训练”的 hard-target baseline |
| **Fixed Teacher** | 使用更大尺寸的预训练模型作为固定 teacher（不更新） |
| **CHAIRi/s** | Caption Hallucination Assessment，instance/sentence 级别幻觉率（↓越好） |
| **POPE** | Polling-based Object-level Pretraining Evaluation，对象幻觉检测（↑越好） |
| **AMBER** | Hallucination Benchmark，综合幻觉评估（Generative + Discriminative） |
| **MMStar** | Vision-Language Understanding 基准（↑越好） |
| **CV-Bench** | Visual Reasoning 基准（↑越好） |

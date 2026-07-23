# Model-Delta Logprob Probe 写作建议

> 重要修正：本文档记录的是早期粗粒度 model-delta 结果。主文机制分析不要直接照搬 “Frozen RKL selected/unselected” 的说法，因为 Frozen RKL 训练时对所有有效 response tokens 施加 RKL，没有 selected token。若在 Frozen RKL 上按 RiskMask score 分组，只能称为 `offline risk tail` / `offline safe bulk`，即事后诊断分层。更完整的写作口径见 [model_delta_surgery_probe_writing_notes_zh.md](model_delta_surgery_probe_writing_notes_zh.md)。

这份文档专门整理 `res-opd/probes/results/model_delta_logprob` 下三组 8B `tr=0.75` probe 结果，并说明它们应该如何接入当前论文的 `res-opd/paper/sections/analysis.tex`。

当前 Overleaf 里的 Analysis 结构已经比较明确：

1. 训练前 dual-view probe：说明 low-resolution teacher view 在训练前已经富集 hallucination-prone object mentions。
2. RiskMask top-30% vs random top-30%：说明 RiskMask 不是随机稀疏化。
3. Final object decomposition：说明 RiskMask 最终不是简单少说，而是净减少 hallucinated objects，同时保留 correct objects。

新的 model-delta probe 不应该替换上面三段，而应该补成第四个机制证据：**训练后的模型到底怎样改变 base model 的 token distribution**。它回答的是“RiskMask 是不是更局部、更保守地 reshape/suppress 高风险 token，而不是全局粗暴改变语言分布”。

## 1. 三组实验分别是什么

结果目录：

- `res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_base_self__tr0p75/`
- `res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_rkl_tr075_s312__tr0p75/`
- `res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_riskmask_tr075_s100__tr0p75/`

共同设置：

- Caption source：8B Base model 在原图上的 caption。
- Base distribution：8B Base model + 原图，对 base caption 做 forced scoring。
- Low-res teacher gate：同一个 8B Base model + `tr=0.75` low-res 图，用来计算 `RKL(base || low-res teacher)`，这是 RiskMask 的训练前 gate signal。
- After distribution：
  - `base_self`：after 仍然是 Base 自己，用作 sanity check。
  - `rkl_tr075_s312`：Frozen RKL checkpoint，对应主表 8B Frozen RKL 行，`CHAIRi=0.2894, CHAIRs=0.0702, ObjF1=0.7258`。
  - `riskmask_tr075_s100`：RiskMask p30 checkpoint，对应主表 8B Res-OPD 行，`CHAIRi=0.2873, ObjF1=0.7272`。

核心约定：

- `RKL(after || base)`、`KL(base || after)`、`JSD(after, base)` 衡量训练后模型相对 base 的真实分布漂移。
- `RKL(base || low-res teacher)` 是训练前 cross-view gate signal，不是训练后的漂移。
- `delta_logp = log p_after(y_t) - log p_base(y_t)`，衡量训练后模型是否提高或降低 base caption 中当前 emitted token 的概率。
- `student entropy` / `base entropy` 是当前位置 `p(. | image, prefix)` 的词表分布熵，不是下一个独立词的熵；它和当前 emitted token `y_t` 的位置对齐。

## 2. Base-self sanity check

`base_self` 的作用是排除工具链假信号：同一个 base model、同一张原图、同一条 base caption 做两次 scoring 时，训练后漂移指标应该为 0。

关键数字：

| Token group | RKL(after,base) | JSD(after,base) | Delta logp | Top1 same | TopK Jaccard |
|---|---:|---:|---:|---:|---:|
| correct_object | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| hallucinated_object | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |
| non_object_token | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 |

同时，base-vs-lowres gate 仍然有 hallucination enrichment：

| Object mention | Gate RKL | Base NLL | Base entropy |
|---|---:|---:|---:|
| correct_object | 0.0369 | 0.2739 | 0.4725 |
| hallucinated_object | 0.0576 | 0.4444 | 0.6715 |

写作意义：

- 这组结果说明 model-delta probe 本身没有把 numerical noise 当成分布变化。
- 更重要的是，即使 after=base，low-res gate 依然能区分 hallucinated vs correct mentions；这和 Analysis 第一段的 base dual-view 结论一致。

正文不一定要放这张表。更适合在一句话或脚注里写：

> As a sanity check, scoring the same base captions with the base model twice yields zero after-base divergence and identical top-k sets, while the base-vs-low-resolution gate remains hallucination-enriched.

## 3. Frozen RKL vs RiskMask：训练后漂移更集中在哪里

这张表最适合放进 Analysis 正文或 appendix，回答“训练改变的是不是集中在 high-risk tail”。

| Method | Group | RKL(after,base) | JSD(after,base) | Delta logp | Top1 same | TopK Jaccard | Reshape | Suppress |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen RKL | RiskMask-selected | 0.0610 | 0.0145 | +0.0488 | 0.8407 | 0.8881 | 0.1604 | 0.2155 |
| Frozen RKL | Unselected | 0.0067 | 0.0015 | -0.0015 | 0.9934 | 0.9139 | 0.0071 | 0.0716 |
| RiskMask | RiskMask-selected | 0.0453 | 0.0109 | +0.0402 | 0.8600 | 0.9017 | 0.1406 | 0.2171 |
| RiskMask | Unselected | 0.0052 | 0.0012 | -0.0024 | 0.9940 | 0.9232 | 0.0063 | 0.0687 |

可写结论：

- 两种训练都主要改变 RiskMask-selected tail，而不是均匀改写所有 token。
- Frozen RKL 中 selected token 的 JSD 是 unselected 的约 `9.7x`，RiskMask 中 selected token 的 JSD 是 unselected 的约 `9.1x`。
- RiskMask 相比 Frozen RKL 更保守：selected token 的 JSD 从 `0.0145` 降到 `0.0109`，RKL(after,base) 从 `0.0610` 降到 `0.0453`，Top1 same 从 `0.8407` 升到 `0.8600`，TopK Jaccard 从 `0.8881` 升到 `0.9017`。
- 但 RiskMask 没有失去抑制能力：selected token 的 suppress rate 为 `0.2171`，与 Frozen RKL 的 `0.2155` 基本相当。

推荐主文句子：

> The learned distribution shift is highly localized. Under RiskMask, selected tokens have about nine times larger JSD from the base model than unselected tokens, while unselected tokens remain almost unchanged. Compared with uniform RKL, RiskMask produces smaller overall drift and higher top-k overlap, but keeps a similar suppression rate on selected tokens, suggesting a more conservative reshape of the high-risk tail.

中文口径：

> 训练后的分布漂移高度局部化。RiskMask 选中的 token 相比未选 token 有约 9 倍的 JSD，而未选 token 基本保持 base 分布。相比 uniform RKL，RiskMask 的整体漂移更小、top-k 重合度更高，但 selected token 上的 suppress rate 与 RKL 接近，说明它更像是对高风险 tail 做保守 reshape，而不是全局改写模型。

## 4. Object-token 层面：RiskMask 更保守地保留 correct object

这张表回答“RiskMask 为什么比 RKL 更稳”：它没有在 object token 上制造更大的全局扰动。

| Method | Object type | RKL(after,base) | JSD(after,base) | Delta logp | Delta entropy | Top1 same | TopK Jaccard | Suppress |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen RKL | correct_object | 0.0233 | 0.0055 | +0.0167 | -0.0114 | 0.9579 | 0.8951 | 0.0830 |
| Frozen RKL | hallucinated_object | 0.0415 | 0.0094 | +0.0116 | +0.0044 | 0.9359 | 0.8904 | 0.1417 |
| RiskMask | correct_object | 0.0175 | 0.0042 | +0.0132 | -0.0047 | 0.9647 | 0.9072 | 0.0857 |
| RiskMask | hallucinated_object | 0.0292 | 0.0069 | +0.0056 | +0.0127 | 0.9481 | 0.9011 | 0.1478 |

可写结论：

- Hallucinated object tokens 的 after-base drift 高于 correct object tokens，这说明训练确实更强地作用在幻觉相关 object token 上。
- RiskMask 相比 Frozen RKL 的漂移更小：correct-object JSD `0.0042 vs 0.0055`，hallucinated-object JSD `0.0069 vs 0.0094`。
- RiskMask 的 top-k overlap 更高：correct `0.9072 vs 0.8951`，hallucinated `0.9011 vs 0.8904`。
- RiskMask 对 hallucinated object 的 suppress rate 稍高：`0.1478 vs 0.1417`，同时对 correct object 的 suppress rate 基本相近：`0.0857 vs 0.0830`。

推荐正文不要强调 “delta logp 均值全是正”。forced scoring 固定在 base caption 上，delta logp 的均值为正只能说明模型对很多 base tokens 更熟悉/更 confident；它不能直接等价于“训练会生成更多幻觉”。更强的证据是：

- final object decomposition 里 RiskMask 净减少 hallucinated objects；
- mention-fate 表里被移除 hallucination 在训练前 gate RKL/NLL 显著更高；
- qualitative examples 中高风险 hallucination token 被明显压低。

## 5. Mention fate：被移除 hallucination 在训练前就是高风险

这张表最适合放 appendix，或作为 Analysis 正文一两句话的数字支撑。

| Method | Fate | N | Gate RKL | Base NLL | Delta logp | JSD(after,base) |
|---|---|---:|---:|---:|---:|---:|
| Frozen RKL | kept_correct | 13315 | 0.0362 | 0.2702 | +0.0174 | 0.0058 |
| Frozen RKL | kept_hallucinated | 2253 | 0.0474 | 0.3805 | +0.0107 | 0.0073 |
| Frozen RKL | removed_correct | 154 | 0.1010 | 0.5884 | +0.0332 | 0.0118 |
| Frozen RKL | removed_hallucinated | 489 | 0.1045 | 0.7390 | +0.0031 | 0.0161 |
| RiskMask | kept_correct | 13332 | 0.0366 | 0.2707 | +0.0152 | 0.0044 |
| RiskMask | kept_hallucinated | 2260 | 0.0482 | 0.3850 | +0.0055 | 0.0057 |
| RiskMask | removed_correct | 137 | 0.0729 | 0.5783 | -0.0153 | 0.0101 |
| RiskMask | removed_hallucinated | 482 | 0.1014 | 0.7231 | -0.0030 | 0.0102 |

最值得写的点：

- 对 RiskMask 来说，被移除 hallucination 的训练前 gate RKL 是 kept hallucination 的 `2.10x`：`0.1014 / 0.0482`。
- 被移除 hallucination 的 base NLL 是 kept hallucination 的 `1.88x`：`0.7231 / 0.3850`。
- RiskMask 删除的 correct mentions 更少：`137`，Frozen RKL 为 `154`。
- RiskMask 删除的 hallucinated mentions 与 Frozen RKL 相近：`482` vs `489`。

推荐表述：

> Mentions that disappear after RiskMask are already high-risk under the pre-training gate: removed hallucinations have more than twice the gate RKL of kept hallucinations and nearly twice the base NLL. Compared with uniform RKL, RiskMask removes fewer correct mentions while removing a comparable number of hallucinated mentions.

中文口径：

> RiskMask 删除掉的 hallucination 在训练前就已经是高风险区域：它们的 gate RKL 是保留下来的 hallucination 的两倍以上，base NLL 接近两倍。相比 uniform RKL，RiskMask 删除的 correct mentions 更少，同时删除的 hallucinated mentions 数量接近。这是 RiskMask “更保守但不弱化 hallucination suppression” 的关键证据。

## 6. 当前 Analysis.tex 应该怎么改

现在的 `analysis.tex` 已有三段，不建议大改。建议做两个局部增强：

### 6.1 在 RiskMask enrichment 表后加入一段 model-delta 解释

位置：当前 `Table~\ref{tab:analysis_enrichment}` 之后，object decomposition 段之前。

建议新增一个小表，标题类似：

`Distribution shift concentrates on RiskMask-selected tokens`

列：

- Method
- Token group
- JSD(after,base)
- RKL(after,base)
- Top1 same
- TopK Jaccard
- Suppress
- Reshape

行只放 4 行：Frozen RKL selected/unselected，RiskMask selected/unselected。不要放 object-token 大表，否则主文会太挤。

建议新增正文要点：

- selected vs unselected 的漂移差距约 9 倍；
- RiskMask 比 Frozen RKL drift 更小、top-k overlap 更高；
- selected token 上 suppress rate 与 Frozen RKL 相当。

### 6.2 在 object decomposition 段里补一句 mention-fate

位置：当前 object decomposition 段最后一句前后。

建议补一句：

> This behavior is consistent with the forced-scoring fate analysis: hallucinated mentions removed by RiskMask have much larger pre-training gate RKL and base NLL than hallucinated mentions that remain, while RiskMask removes fewer correct mentions than uniform RKL.

不要把 mention-fate 表放主文，除非版面足够。它放 appendix 更合适。

## 7. 哪些东西不要写过头

不要写：

- “RiskMask 直接降低所有 hallucinated token 的 logprob。”
- “NLL 比 entropy 离线选择永远更好。”
- “RiskMask 是一个 hallucination classifier。”
- “RKL/JSD/KL 三者实验显示 RKL 显著优于 JSD/KL。”目前没有必要做这个 claim；方法目标就是 RKL 训练，分析中选择 RKL 作为 gate 是 objective-aligned。

建议写：

- “RiskMask is an OPD-aligned gate, not a standalone hallucination detector.”
- “Low-resolution disagreement and emitted-token uncertainty identify a high-risk tail.”
- “RiskMask localizes the distribution shift and preserves high-overlap regions.”
- “The final caption changes show net hallucination reduction without net loss of correct object mentions.”

## 8. 给论文 AI 的指令

可以把下面这段直接发给论文 AI：

```text
请只修改 `res-opd/paper/sections/analysis.tex`，不要改 experiments/method/introduction。先阅读 `res-opd/plan/model_delta_logprob_probe_writing_notes_zh.md`，然后基于其中的三组 model-delta probe 结果，在当前 Analysis 的第二段和第三段之间加入一个新的机制段落，解释训练后模型相对 base caption 的 distribution shift。

写作目标：
1. 保持当前 Analysis 的三段结构与已有表格，不要重写整节。
2. 新增一个小表，建议标题为 “Distribution shift concentrates on RiskMask-selected tokens”。表中只放四行：Frozen RKL selected/unselected、RiskMask selected/unselected；列包含 JSD(after,base)、RKL(after,base)、Top1 same、TopK Jaccard、Suppress、Reshape。
3. 正文重点写：selected token 的分布漂移约为 unselected token 的 9 倍；RiskMask 相比 Frozen RKL 有更小的 JSD/RKL 漂移、更高 top-k overlap，但 selected token 的 suppress rate 基本相当，说明 RiskMask 是更保守、更局部的 high-risk tail reshape。
4. 在 object decomposition 段补一句 mention-fate 证据：RiskMask 删除的 hallucinated mentions 在训练前有更高 gate RKL 和 base NLL；相比 Frozen RKL，RiskMask 删除更少 correct mentions，同时删除相近数量的 hallucinated mentions。
5. 不要声称 RiskMask 直接降低所有 hallucinated token 的 logprob；forced scoring 下 delta logp 均值不适合这样解释。不要声称 NLL 是所有 offline selector 中最强；RiskMask 的优势应写成 OPD-aligned training gate，而不是 hallucination classifier。

请保留 AAAI 风格的简洁写法，英文正文为主，中文注释可以按当前 analysis.tex 的风格保留在 `% 中文：...` 后面。
```

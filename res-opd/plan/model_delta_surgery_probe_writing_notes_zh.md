# Model-Delta Surgery Probe 写作建议

这份文档专门给 `res-opd/paper/sections/analysis.tex` 使用，目标是补强一个问题：

> RiskMask 到底是在“有问题的 token”上做有针对性的局部修改，还是只是因为只训练 top-30% token，所以整体约束更轻？

旧的 `model_delta_logprob_probe_writing_notes_zh.md` 只能作为粗粒度 sanity check，不适合作为主机制证据。尤其要注意：**Frozen RKL 没有 selected token**。如果在 Frozen RKL checkpoint 上按 RiskMask score 分出 selected/unselected，那只是 `offline risk tail` / `offline safe bulk` 的事后分层，用于和 RiskMask 对齐比较“哪些 token 本来会被 RiskMask 选中”，不能写成 Frozen RKL 训练时真的只作用在 selected token 上。

## 1. 新 probe 应回答的问题

主文里的机制问题应该拆成四个层次：

1. **风险信号来自哪里**：训练前 `base(original)` vs `base(low-res teacher)` 的 RKL 是否富集 hallucinated object mentions。
2. **RiskMask 选的是不是非随机 token**：`rank(RKL) × rank(NLL)` 的 top-p 是否比 random top-p 更富集幻觉对象。
3. **训练后到底改了哪里**：训练后 checkpoint 相对 base 的分布漂移是否集中在 `high-RKL/high-NLL` 区域，而不是全局改写。
4. **最终 caption 变好了是不是因为“少说”**：被删除的 hallucinated mentions 是否更多落在高风险区，同时 correct mentions 是否被保留或轻微 sharpen。

第 1、2、4 点已经分别由 `base_dual_view_probe_results.md`、`probe_analysis_writing_notes_zh.md`、caption object decomposition 支撑。现在缺的是第 3 点：**token distribution surgery**。

## 2. 关键定义

### 2.1 三个分布

对同一条 base caption 做 forced scoring：

- `base`: Base model + original image。
- `teacher`: 同一个 Base model + low-res image，默认 `tr=0.75`。
- `after`: 训练后的 checkpoint + original image。

其中：

- `RKL(base || teacher)` 是训练前 cross-view gate signal，用来判断 token 是否处于低分辨率分歧区。
- `base NLL` 是 base model 对自己已经生成的当前 token 的负对数概率，用来判断这个 emitted token 是否低置信。
- `RKL(after || base)`、`JSD(after, base)`、`delta_logp(after-base)`、`top-k overlap` 才是训练后模型相对 base 的真实变化。

### 2.2 RiskMask 的事后分层

对所有方法都可以事后计算：

```text
offline_risk_score = rank(RKL(base || low-res teacher)) × rank(base NLL)
offline_risk_tail = top-p tokens by offline_risk_score
offline_safe_bulk = remaining tokens
```

但写作时必须区分：

- **RiskMask checkpoint**：`offline_risk_tail` 近似训练中真实参与 RKL 的 token。
- **Frozen RKL checkpoint**：`offline_risk_tail` 只是为了分析方便做的事后分层；Frozen RKL 实际对所有有效 response tokens 施加 RKL。

### 2.3 Token / mention 类型

建议至少统计这些类型：

| 类型 | 定义 | 论文里想证明什么 |
|---|---|---|
| `correct_object` | base caption 中命中的 GT object mention | RiskMask 不能粗暴压掉正确对象 |
| `hallucinated_object` | base caption 中不在 GT object 集合里的 object mention | RiskMask 应优先压制 unsupported object |
| `non_object_token` | 不属于 object mention 的普通 token | 检查 mask 是否主要改变对象相关区域，还是只是在改语言风格 |
| `correct_highrisk_uncertain` | correct object 且 high RKL/high NLL | 关键保护区：如果这些还能保留，说明不是简单“高不确定就删除” |
| `hallucination_highrisk_uncertain` | hallucinated object 且 high RKL/high NLL | 关键打击区：应该有更高 remove/suppress/reshape |
| `hallucination_highrisk_confident` | hallucinated object 且 high RKL/low NLL | 可能是难例：模型很自信但 low-res view 反对 |
| `hallucination_lowrisk_uncertain` | hallucinated object 且 low RKL/high NLL | 可能是语言不确定，但低分辨率 teacher 没提供明确反证 |

## 3. 新增脚本

新增脚本：

```text
res-opd/probes/analyze_model_delta_surgery.py
```

它不做模型 forward，只读取已经生成的：

```text
model_delta_trace.jsonl
model_delta_summary.json
```

输出：

```text
surgery_summary.json
surgery_summary.md
surgery_quadrants.csv
surgery_examples.jsonl
```

其中 `surgery_quadrants.csv` 用于画图，`surgery_examples.jsonl` 用于挑 qualitative examples。examples 里会带：

- `image_id`
- `image_path`
- `canonical_object`
- `object_type`
- `after_change_status`
- `context`
- `base_caption`
- `after_caption`
- `rkl_base_to_teacher_mean`
- `base_nll_mean`
- `teacher_minus_base_nll_mean`
- `jsd_after_base_mean`
- `delta_logp_after_base_mean`
- `mechanism_*_frac`

## 4. 建议看的核心表

### 4.1 `token_role_x_rkl_nll`

用途：回答 “训练后的漂移是否集中在 high-RKL/high-NLL token”。

推荐画图：

- x 轴：`lowRKL/lowNLL`, `lowRKL/highNLL`, `highRKL/lowNLL`, `highRKL/highNLL`
- y 轴：`correct_object`, `hallucinated_object`, `non_object_token`
- color：`JSD(after, base)` 或 `suppress_emitted_rate`
- facet：Frozen RKL vs RiskMask

期待口径：

- 如果 RiskMask 的 `highRKL/highNLL` 区域有明显更高的 drift/suppress，而 `lowRKL/lowNLL` 区域低漂移、高 top-k overlap，就能说明它不是普通稀疏化，而是局部作用。
- 如果 `non_object_token` 也大面积高漂移，需要弱化“对象专属 detector”的说法，改写成“token-level risk gate that is enriched for hallucinated objects”。

### 4.2 `mention_fate_x_rkl_nll`

用途：回答 “最终被删除的 hallucinated mentions 是否训练前就更高风险”。

重点比较：

- `kept_hallucinated` vs `removed_hallucinated`
- `kept_correct` vs `removed_correct`
- RiskMask vs Frozen RKL

最有利的写法：

> Removed hallucinated mentions concentrate in the high-RKL/high-NLL bucket, while kept correct mentions are dominated by low-risk buckets. Compared with Frozen RKL, RiskMask removes a comparable number of hallucinated mentions but fewer correct mentions.

中文口径：

> 被 RiskMask 删除的幻觉对象在训练前就集中在 high-RKL/high-NLL 区域；而被保留的正确对象主要处于 low-risk 区域。相比 Frozen RKL，RiskMask 删除的幻觉对象数量接近或更多，但误删 correct mention 更少，因此不是简单“少说”，而是更有选择性。

### 4.3 `mention_fate_x_offline_risk_bin`

用途：最直接地支持“RiskMask top-p 是有针对性的”。

重点看：

- `removed_hallucinated/offline_risk_tail` 占比是否高。
- `removed_correct/offline_risk_tail` 是否低于 Frozen RKL。
- `kept_correct/offline_safe_bulk` 是否稳定。

这张表适合做 appendix 表，正文可只引用一句。

### 4.4 `token_role_x_teacher_stance_nll`

用途：检查 teacher 是否在“反对”某些 hallucinated token。

这里的 teacher stance 定义为：

```text
teacher_minus_base_nll = teacher_nll - base_nll
```

- `teacher_rejects`: teacher 对当前 emitted token 的 NLL 明显高于 base。
- `teacher_supports`: teacher 对当前 emitted token 的 NLL 明显低于 base。
- `teacher_neutral`: 中间区域。

建议口径：

- 如果 `hallucinated_object/teacher_rejects/highNLL` 的 suppress/removed rate 更高，可以写 low-res teacher 提供了 conservative counterfactual rejection。
- 如果 teacher uncertainty 很高导致误伤小物体，则写成 limitation：低分辨率 teacher 对小/细粒度对象可能不稳定，这也是为什么 RiskMask 不能只依赖 teacher uncertainty。

## 5. 机制标签怎么解释

脚本会给 token 打一个粗粒度 `mechanism`：

| Mechanism | 直观含义 | 正文里怎么用 |
|---|---|---|
| `stable` | after 和 base 分布几乎不变 | 说明低风险区域被保留 |
| `sharpen_same_top1` | top-1 不变，entropy/top1 margin 更确定 | 支撑“正确但不确定 token 被 sharpen” |
| `reshape_topk` | top-1 或 top-k 支持集合变化明显 | 支撑“高风险 token 被重排” |
| `suppress_emitted` | base emitted token 的 logprob 下降 | 支撑“幻觉 token 被压低” |
| `boost_emitted` | base emitted token 的 logprob 上升 | 不能简单说好坏，要结合 object fate |

注意：`delta_logp` 是 forced scoring 在 base caption token 上的变化。它不能单独解释生成时最终对象增减，必须和 `after_change_status` 或 final object decomposition 一起用。

## 6. 最适合主文的图

建议主文放一个 2-panel figure，而不是再塞很多均值表。

### Panel A：Risk landscape heatmap

数据源：`surgery_quadrants.csv` 中 `token_role_x_rkl_nll`。

画法：

- 每个 cell 是一个 token bucket：object role × RKL bin × NLL bin。
- color 用 `suppress_emitted_rate` 或 `jsd_after_base`。
- 两列分别画 Frozen RKL 和 RiskMask。

想表达：

> RiskMask 的训练后漂移更集中在 high-RKL/high-NLL 的 hallucinated object 区域，而 low-risk correct/object 和 non-object 区域保持更稳定。

### Panel B：Mention fate by risk bucket

数据源：`mention_fate_x_offline_risk_bin` 或 `focus_summary`。

画法：

- stacked bar：`kept_correct`, `removed_correct`, `kept_hallucinated`, `removed_hallucinated`
- 按 `offline_risk_tail` vs `offline_safe_bulk` 分组。

想表达：

> RiskMask 不是简单减少 object mentions；它优先影响训练前已呈现 cross-view disagreement + low confidence 的 mention。

### Panel C 可放 appendix：qualitative examples

数据源：`surgery_examples.jsonl`。

优先挑：

- `removed_hallucination_highrisk_suppressed`
- `kept_correct_lowrisk_stable`
- `uncertain_correct_preservation_or_damage`
- `failure_removed_correct`

每个例子展示：

```text
Image
Base caption
RiskMask caption
Highlighted object mention
RKL/NLL/teacher stance/delta_logp
```

## 7. 写作口径：如何凸显 RiskMask

主线不要写成 “RiskMask 的总 drift 更小”。这太弱，而且审稿人会说 top-30% mask 本来就更轻。

应该写成：

1. **Risk signal before training**：hallucinated object mentions 在 low-res disagreement 和 uncertainty 上显著更高。
2. **Targeted action during training**：RiskMask 只把 RKL 放到 `high-RKL × high-NLL` 的 tail，而不是对所有 token 平均施压。
3. **Local distribution surgery after training**：训练后的主要分布改变集中在 high-risk 区域；低风险 correct tokens 的 top-k overlap 高、JSD 低。
4. **Object-level outcome**：最终 caption 中 hallucinated objects 净减少，correct objects 不减少甚至略增。

一句话版本：

> RiskMask is not merely a weaker RKL objective. It uses low-resolution disagreement to locate visually fragile tokens and emitted-token NLL to avoid over-regularizing confident regions, resulting in localized distribution surgery: high-risk hallucinated mentions are more likely to be reshaped or removed, while low-risk correct mentions remain close to the base model.

中文：

> RiskMask 不是简单削弱版 RKL。它用低分辨率分歧定位视觉证据脆弱的 token，再用当前 emitted token 的 NLL 避免过度正则化高置信区域，从而形成局部的 distribution surgery：高风险 hallucinated mentions 更容易被 reshape 或删除，而低风险 correct mentions 保持接近 base model。

## 8. 不利结果怎么处理

如果 entropy-only 在某些离线 enrichment 指标上比 NLL 更强，不要硬说 NLL 是最强 detector。更好的口径是：

- entropy 证明 student uncertainty 本身有效；
- NLL 是更贴合 OPD loss 的 emitted-token confidence，因为 RKL 训练也是沿当前生成 token 的 trajectory 发生；
- RiskMask 的优势不是“离线分类最强”，而是“训练后 caption 结果最好/更稳”，尤其体现在 CHAIR、AMBER generative、object decomposition 或 POPE guardrail。

如果 high-risk correct mentions 也被改动：

- 不要说 RiskMask 完全保护所有 correct objects。
- 改写成 “mitigates over-regularization compared with Frozen RKL”，并用 `removed_correct` 数量更少或 `kept_correct` top-k overlap 更高支撑。

如果 confident hallucinations 仍然保留：

- 写成 limitation：NLL-based gate 主要针对“不确定 + 跨视角分歧”的幻觉，强语言先验导致的 confident hallucination 需要 teacher-rejection 或 external verifier。

## 9. 服务器运行命令

这条命令只做二次统计，不重新跑模型：

```bash
tmux new-session -d -s probe_8b_model_delta_surgery 'cd /home/liuyanlin.lyl/notebook/lyl/opd/Vision-OPD && mkdir -p res-opd/logs && /home/liuyanlin.lyl/.conda/envs/vision-opd/bin/python3 res-opd/probes/analyze_model_delta_surgery.py --run base_self=res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_base_self__tr0p75 --run frozen_rkl_s312=res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_rkl_tr075_s312__tr0p75 --run riskmask_p30_s100=res-opd/probes/results/model_delta_logprob/Qwen3VL-8B-Instruct__8b_riskmask_tr075_s100__tr0p75 --output-dir res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask --top-p 0.30 --high-frac 0.30 2>&1 | tee res-opd/logs/probe_8b_model_delta_surgery.log'
```

运行后检查：

```bash
ls -lh res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask
sed -n '1,220p' res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_summary.md
head -n 5 res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_examples.jsonl
```

如果报 `Missing model_delta_trace.jsonl`，说明 trace 目录名和这里写的不一致；让服务器 AI 用下面命令定位：

```bash
find res-opd/probes/results/model_delta_logprob -name model_delta_trace.jsonl -print
```

## 10. 给论文 AI 的指令

等 `surgery_summary.md` 和 `surgery_examples.jsonl` 生成后，把下面这段发给论文 AI：

```text
请只修改 `res-opd/paper/sections/analysis.tex`，不要改 experiments/method/introduction。先阅读这些文件：
1. `res-opd/plan/analysis_section_plan_zh.md`
2. `res-opd/plan/probe_analysis_writing_notes_zh.md`
3. `res-opd/plan/model_delta_surgery_probe_writing_notes_zh.md`
4. `res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_summary.md`
5. `res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_examples.jsonl`

现在重写 Analysis 中关于 model-delta / distribution shift 的段落。不要再使用“Frozen RKL selected token”这种表述；Frozen RKL 的 selected/unselected 只能叫 offline risk tail / offline safe bulk，是事后分层，不是训练 mask。

写作目标：
1. 把机制写成 targeted distribution surgery，而不是“RiskMask 总体漂移更小”。
2. 用 surgery_summary 的 quadrant/fate 统计说明：RiskMask 的改动集中在 high-RKL/high-NLL 区域；low-risk correct mentions 保持更高 top-k overlap / 更低 JSD；removed hallucinated mentions 在训练前具有更高 gate RKL 和 NLL。
3. 明确区分 correct object、hallucinated object 和 non-object token；如果 non-object 也有变化，不要把 RiskMask 写成 object detector，而写成 OPD-aligned token-level gate enriched for hallucinated object mentions。
4. 加一个 figure/table 建议：Panel A 用 `token_role_x_rkl_nll` heatmap，Panel B 用 `mention_fate_x_offline_risk_bin` stacked bar，qualitative examples 从 `surgery_examples.jsonl` 里挑。
5. 不要声称 NLL 离线富集一定强于 entropy；可以写 entropy 也说明 uncertainty 有用，而本文采用 NLL 是因为它直接对应当前 emitted token 的 confidence，更贴合 on-policy RKL 训练。
6. 风格保持 AAAI/顶会正文：少堆数字，多解释机制；中文注释可以保留在 `% 中文：...` 后面。
```

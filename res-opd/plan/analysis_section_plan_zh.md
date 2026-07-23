# Analysis 部分写作规划

这份规划面向 `paper/sections/analysis.tex`。目标不是再堆 benchmark，而是解释为什么 Res-OPD 的设计合理：低分辨率 teacher 产生的分布分歧是否真的指向 hallucinated object mentions，RiskMask 为什么要结合 cross-view disagreement 和 student uncertainty，以及这种训练到底删掉了哪些对象。

## 建议正文结构

### 1. Cross-resolution disagreement highlights hallucination-prone object mentions

主问题：低分辨率 teacher 不是更强 oracle，而是一个 conservative counterfactual view；如果 object mention 依赖弱视觉证据或语言先验，原图 student 与低分辨率 teacher 的分布会更容易分歧。

可用数据：
- 首选主文数据：`res-opd/plan/base_dual_view_probe_results.md`
- Caption source = 8B Base model original-image output。
- Student = 同一个 8B Base model + original image；Teacher = 同一个 8B Base model + low-res `tr=0.75` image。
- 这组 probe 没有用训练后的 RiskMask checkpoint，也没有重新生成 caption，因此最适合作为 RiskMask 设计动机证据。
- Correct object mentions: count 13,469, RKL mean 0.0369, JSD mean 0.0079, student NLL 0.2739, entropy 0.4725。
- Hallucinated object mentions: count 2,742, RKL mean 0.0576, JSD mean 0.0126, student NLL 0.4444, entropy 0.6715。
- Hallucinated objects have +56% RKL, +62% NLL, +42% entropy relative to correct objects。
- Top 10% RKL bucket: hallucination precision 25.15%, recall 14.88%, lift 1.49x。
- Top 10% student NLL bucket: hallucination precision 27.44%, recall 16.23%, lift 1.62x。
- Top 10% student entropy bucket: hallucination precision 29.90%, recall 17.69%, lift 1.77x。
- RiskMask-NLL top 30%: hallucination precision 23.27%, recall 41.28%, lift 1.38x；Random top 30%: precision 17.43%, recall 30.93%, lift 1.03x。

推荐正文表：`Hallucination enrichment in high-risk object mentions`

| Signal | Top frac | Precision | Recall | Lift |
|---|---:|---:|---:|---:|
| RKL full vs low-res | 10% | 25.15 | 14.88 | 1.49 |
| Student NLL | 10% | 27.44 | 16.23 | 1.62 |
| Student entropy | 10% | 29.90 | 17.69 | 1.77 |
| Random | 30% | 17.43 | 30.93 | 1.03 |
| RiskMask-NLL | 30% | 23.27 | 41.28 | 1.38 |

写作结论：RKL、NLL、entropy 都不是完美 hallucination detector，但高分区间显著富集 hallucinated mentions。这正好支持 RiskMask 的设计：用 cross-view disagreement 捕捉视觉证据脆弱性，再用 student uncertainty 避免只按分辨率差异惩罚所有对象。

注意：entropy 在这组 offline enrichment 上略强于 NLL，因此正文不要写“NLL 是最强 detector”。更稳的口径是：NLL 与 entropy 都说明 student uncertainty 有用；本文采用 emitted-token NLL，是因为它直接绑定当前生成 token 的置信度，更贴合训练中“这个 object token 是否应该承受 RKL 约束”的 gate 目标。Entropy-only 和 RiskEntropy 放 appendix / ablation。

### 2. Low-resolution views provide more useful disagreement than same-image regularization

主问题：如果 `tr=1.0` 原图 RKL 也能有一点效果，怎么证明低分辨率 teacher 不是多余的？

可用数据：
- 8B no-resolution-gap control: `res-opd/plan/instruct_tr10_eval_results.md`，teacher ratio = 1.0，teacher 和 student 都看原图。
  - Base: CHAIRi 0.3025, ObjF1 0.7178, AMBER CHAIR 6.0, MMStar 0.6480, CV-Bench 0.8701。
  - Best RKL by CHAIRi: step100, CHAIRi 0.2903, ObjF1 0.7234。
  - Best RiskMask by CHAIRi: step50, CHAIRi 0.2904, ObjF1 0.7243。
  - `tr=1.0` 能带来 same-image consistency regularization，但 RiskMask 相比 RKL 没有稳定拉开；这和 `tr=0.75` 下 RiskMask 更强的 object-decomposition 证据形成对比。
- Duplicate base original-image sanity: `1_duplicate_base_sr10_summary.json`，correct/hallucinated RKL 都是 0，JSD 约 2.6e-9，说明相同模型相同原图的重复 forward 本身没有 KL 信号。
- Same-image trained-student vs frozen-base teacher: `5_dual_model_tr10_captions_summary.json`，correct RKL 0.0038，hallucinated RKL 0.0047，top 10% RKL precision 15.9%，lift 1.14x。
- Full-resolution vs low-resolution view: `3_dual_view_full_vs_lowres_summary.json`，correct RKL 0.0375，hallucinated RKL 0.0524，top 10% RKL precision 20.8%，lift 1.47x。
- 8B Base full-resolution vs low-resolution view: `base_dual_view_probe_results.md`，correct RKL 0.0369，hallucinated RKL 0.0576，top 10% RKL precision 25.15%，lift 1.49x。

推荐正文表：`Where does the risk signal come from?`

| Probe | Correct RKL | Hallucinated RKL | Top-10 RKL Precision | Lift |
|---|---:|---:|---:|---:|
| Duplicate original view | 0.0000 | 0.0000 | -- | -- |
| Same-image trained student vs teacher | 0.0038 | 0.0047 | 15.9 | 1.14 |
| Original vs low-resolution view | 0.0375 | 0.0524 | 20.8 | 1.47 |
| 8B Base original vs low-resolution view | 0.0369 | 0.0576 | 25.15 | 1.49 |

写作结论：同图 RKL / RiskMask 可以提供 same-image consistency regularization，因此 `tr=1.0` 的 CHAIRi 也会下降；但这不是本文主线的完整解释。低分辨率 view 在训练前就产生更大、更 hallucination-sensitive 的分布差异，并且 `tr=0.75` RiskMask 的 final object decomposition 相比 uniform RKL 更能减少 hallucinated objects、保留 correct objects。正文要把 `tr=1.0` 写成 control：它说明 consistency regularization 有用，但低分辨率 conservative view 提供了更 OPD-specific 的 risk signal。

### 3. Object-level caption changes: what gets removed?

主问题：Res-OPD 是否只是少说对象？还是优先删除不可靠对象，同时尽量保留正确对象？

可用数据：
- 首选主文数据：`res-opd/probes/results/final_8b_tr075_object_decomposition/caption_object_decomposition_summary.{md,json}`
- Base: hallucinated 930, correct 2144, mentioned 3074, CHAIRi 0.3025, ObjF1 0.7186。
- Frozen RKL (`tr=0.75`, step312): hallucinated 876, correct 2151, mentioned 3027, CHAIRi 0.2894, ObjF1 0.7279；paired net hallucinated -54, net correct +7。
- RiskMask (`tr=0.75`, p=0.30, step100): hallucinated 868, correct 2153, mentioned 3021, CHAIRi 0.2873, ObjF1 0.7301；paired net hallucinated -62, net correct +9。

旧 2B/早期 SW decomposition 可作为 appendix 或历史分析，不建议放主文，以免和最终 RiskMask 命名混淆。

推荐正文图/表：
- 正文可以用一张 compact table：Base vs Frozen RKL vs RiskMask，列出 hallucinated objects、correct objects、net hallucination delta、net correct delta。
- 附录放更细的 primary-category decomposition。

关键分类数字：
- RiskMask removed hallucinated = 323, added hallucinated = 261, net hallucinated = -62；removed correct = 131, added correct = 140, net correct = +9。
- Frozen RKL removed hallucinated = 342, added hallucinated = 288, net hallucinated = -54；removed correct = 136, added correct = 143, net correct = +7。
- RiskMask 的 hallucination 净减少主要来自 uncertain/small/background (-31) 和 text/logo/graphic (-25)，ordinary visual object 基本保持稳定 (-2 hallucination, +12 correct)。

写作结论：低分辨率 OPD 主要压制的是视觉证据脆弱、背景小物体、文字/图案诱发的对象提及；RiskMask 相比 uniform RKL 进一步改善 hallucination-coverage trade-off，而不是简单减少 caption 长度或 object mention 数。

### 4. Qualitative examples

主问题：给审稿人直观看到“删掉的是不该说的东西，而不是把 caption 变空”。

可用数据：
- `res-opd/probes/results/examples.jsonl`
- `res-opd/probes/results/full5k_base_vs_variants_v2/caption_object_changes.jsonl`
- `res-opd/probes/results/full5k_base_vs_variants_v2/caption_object_image_changes.jsonl`

建议挑 3 类例子：
- Hallucination removed without correct-object loss：展示去掉 `truck`、`dining table`、`orange` 这类 unsupported object。
- Correct object gained or preserved：展示 caption 仍能提到核心对象，支撑不是 trivial suppression。
- Failure case：展示 low-res/RiskMask 仍会误删细粒度正确对象或新增某些 hallucination，用来让 analysis 更可信。

正文只放 1 个 qualitative figure；更多例子放附录。

### 5. Model-delta surgery: RiskMask 如何局部修改 token distribution

主问题：RiskMask 的优势不能只写成“top-30% mask 后整体漂移更小”。因为只训练 30% token 本来就会让平均分布变化更轻。更强的机制问题是：**RiskMask 是否把训练后的分布改变集中到 high-RKL/high-NLL 的高风险 token，同时让 low-risk correct mentions 保持稳定**。

可用数据：
- 粗粒度记录：`res-opd/plan/model_delta_logprob_probe_writing_notes_zh.md`
- 新的主机制口径：`res-opd/plan/model_delta_surgery_probe_writing_notes_zh.md`
- 服务器生成后应读取：`res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_summary.md`
- 画图数据：`res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_quadrants.csv`
- 例子数据：`res-opd/probes/results/model_delta_logprob_surgery/8b_tr075_base_rkl_riskmask/surgery_examples.jsonl`

推荐主文表达：
- Frozen RKL 没有 selected token；如果按 RiskMask score 分组，只能称为 `offline risk tail`，是事后分析分层。
- RiskMask 的核心 claim 是 **targeted distribution surgery**：它不是简单弱化 RKL，而是利用 `RKL(base||low-res teacher) × base NLL` 找到跨视角不稳定且当前 emitted token 低置信的区域。
- 重点展示 `correct/hallucinated/non-object × high/low RKL × high/low NLL` 的 heatmap，以及 `kept/removed correct/hallucinated × offline risk tail/safe bulk` 的 mention fate 图。
- 如果结果支持，应写：removed hallucinations 更集中在 high-risk 区，kept correct mentions 更集中在 low-risk 区；RiskMask 相比 Frozen RKL 以更少的 safe-region drift 达到相近或更好的 hallucination reduction。
- 如果 entropy-only 某些离线指标更强，不要硬说 NLL 是最强 detector。写成：entropy 证明 uncertainty signal 有用；NLL 作为 emitted-token confidence 更贴近 on-policy RKL loss，因此是本文主方法采用的训练 gate。

## 写作注意

1. Analysis 中可以引用 `res-opd/plan/instruct_tr10_eval_results.md` 里的 8B `tr=1.0` control。表述重点是 same-image/no-resolution-gap OPD 也能带来 consistency regularization，但它不能替代 low-resolution conservative view。
2. Probe 使用的是 object-decomposition pipeline，数值和主 benchmark 表可能略有差异；正文要写成 mechanistic probe，而不是替代主 benchmark。
3. 不要把 high-risk bucket 写成精确 hallucination classifier。更稳的说法是“enriched for hallucinated mentions”。
4. POPE 不适合作为 Analysis 主线；它更适合留在 Experiments 里作为 yes/no hallucination 与通用能力不坍塌的 guardrail。
5. 不要把完整 selector ranking 放进主文第一张 analysis 表。Entropy-only 在 offline enrichment 上可能略强，这只能说明 uncertainty signal 有用；RiskMask 的论文主张应落在“OPD-aligned training gate”和最终训练收益上。
6. 不要把 Frozen RKL 的事后 high-risk 分组写成训练时 selected token。Frozen RKL 是 full-token RKL；只有 RiskMask 才是真正只对 top-p 高风险 token 施加 RKL。

# 当前实验缺失结果汇总

依据当前 `res-opd/paper/sections/experiments.tex` 和 `res-opd/plan/*.md` 的结果文件检查。正文中缺少的数据已经用 `--` 占位，没有用 split accuracy 反推或编造 overall F1。

## P0：当前正文/预留表格直接缺少

| 正文表格 | 行 | 缺少项 | 目前已有数据 | 建议处理 |
|---|---|---|---|---|
| Original-image OPD controls | Qwen3VL-8B-Instruct Frozen RKL, `tr=1.0` | CHAIRi/CHAIRs/ObjF1、POPE Overall F1、AMBER generative CHAIR/HAL、MMStar、CV-Bench | 正文已在 `tab:orig_image_opd_controls` 预留 `--` | 补跑或从服务器找 8B 原图 teacher 的 uniform RKL 结果 |
| Original-image OPD controls | Qwen3VL-8B-Instruct Res-OPD/RiskMask, `tr=1.0` | CHAIRi/CHAIRs/ObjF1、POPE Overall F1、AMBER generative CHAIR/HAL、MMStar、CV-Bench | 正文已在 `tab:orig_image_opd_controls` 预留 `--` | 补跑或从服务器找 8B 原图 teacher 的 RiskMask 结果 |
| Native low-resolution inference controls | 2B Res-OPD, ratio/tr=0.25, step 50 | POPE Overall Accuracy/Recall/F1，正文只需要 F1 | `instruct_tr025_eval_results.md` 只有 `adv/pop/rand Acc = 0.9194/0.9418/0.9527` | 查服务器 raw POPE prediction 或重新跑 overall aggregator |
| Native low-resolution inference controls | 2B Res-OPD, ratio/tr=0.50, step 50 | POPE Overall Accuracy/Recall/F1，正文只需要 F1 | `instruct_tr05_eval_results.md` 只有 `adv/pop/rand Acc = 0.9200/0.9439/0.9583` | 查服务器 raw POPE prediction 或重新跑 overall aggregator |
| Native low-resolution inference controls | 8B Res-OPD, ratio/tr=0.25, step 200 | POPE Overall Accuracy/Recall/F1，正文只需要 F1 | `instruct_tr025_eval_results.md` 只有 `adv/pop/rand Acc = 0.9184/0.9364/0.9533` | 查服务器 raw POPE prediction 或重新跑 overall aggregator |
| Native low-resolution inference controls | 8B Res-OPD, ratio/tr=0.50, step 50 | POPE Overall Accuracy/Recall/F1，正文只需要 F1 | `instruct_tr05_eval_results.md` 只有 `adv/pop/rand Acc = 0.9159/0.9374/0.9621` | 查服务器 raw POPE prediction 或重新跑 overall aggregator |
| General visual capability / Fixed-teacher controls | 4B Base | CV-Bench Overall/2D/3D | 多个 plan 文件里 4B Base CV-Bench 记为 `--` / `—` | 如果保留 4B Base CV 对比，需要补跑或从服务器找 4B base CV-Bench |

说明：POPE 的三个 split accuracy 不能可靠还原 overall F1。F1 需要总体 TP/FP/FN，或者至少需要 overall precision/recall；只有 adv/pop/random 三个 accuracy 不够。

## P1：Analysis 需要补的新机制实验

当前 `paper/sections/analysis.tex` 已经压缩成半页，不再使用旧的 JSD / `global_step_39` probe 表。若后续想让 Analysis 更有数据支撑，建议优先补下面两个轻量机制实验。它们比旧的 divergence probe 更贴近当前主线：`tr=0.75`、RiskMask-NLL、`p=0.30`、最终 selected checkpoint。

| 目标 | 范围 | 需要产出 | 用途 | 优先级 |
|---|---|---|---|---|
| RiskMask top-30% selection enrichment | 至少 Qwen3VL-8B-Instruct 的最终 Res-OPD/RiskMask checkpoint；可选补 2B/4B | 按训练实现里的 `rank(RKL) * rank(NLL)` 计算 token risk score，在 valid response tokens 上取 top 30%；再和 object mention 对齐，报告 selected fraction、hallucination precision、hallucination recall、lift、base hallucination rate | 直接支撑 Analysis 里“为什么 `rho=0.30` 的 high-risk token tail 有意义”，避免用旧的 top-10/top-20 单指标 probe 解释 p30 | 高 |
| Final-checkpoint object-mention decomposition | 至少 Qwen3VL-8B-Instruct：Base / Frozen RKL / Res-OPD(RiskMask), `tr=0.75`；可选补 2B/4B | Mentioned objects、Correct objects、Hallucinated objects、ObjPrecision/ObjRecall/ObjF1、Net hallucinated delta、Net correct delta；可选 primary-category decomposition | 证明 Res-OPD 主要减少 unsupported object mentions，而不是简单少说所有 objects；也能解释为什么 CHAIR/AMBER 改善同时 ObjF1 不崩 | 高 |
| Qualitative examples | 从最终 8B Base / Frozen RKL / Res-OPD 输出中挑 3-4 张图 | 图片、Base caption、RKL caption、Res-OPD caption、removed hallucinated objects、preserved/gained correct objects；至少包含 1 个 failure case | 放正文或附录图，直观说明删掉的是不该说的对象，而不是把 caption 变空 | 中 |

建议实现细节：

- 不要把旧的 JSD probe 继续放正文；JSD 是辅助诊断量，不是方法核心。
- RiskMask selection enrichment 里要用最终训练逻辑：RKL rank 与 NLL rank 相乘，然后取 top 30%。不要再只报告 RKL top-10 或 NLL top-10。
- 如果只能先补一个模型，优先补 8B，因为主表、top-p ablation 和最终叙事都更依赖 8B。
- 如果 object-mention decomposition 的成本较高，先只跑 Base / Frozen RKL / Res-OPD 三行即可；SFT、RandomMask、EntropyMask 可以放到附录或后续补。

## P2：增强说服力但不是当前正文阻塞

当前正文的 native low-resolution inference control 只放了 2B 和 8B，因为 `base_model_eval_results.md` 只有 2B/8B 的 native low-resolution base rows。若希望把 4B 也加入同 ratio 对比，需要：

| 模型 | 推理分辨率 | 需要指标 |
|---|---|---|
| Qwen3VL-4B-Instruct Base | sr=0.25 | CHAIR, POPE Overall + per-split, AMBER generative |
| Qwen3VL-4B-Instruct Base | sr=0.50 | CHAIR, POPE Overall + per-split, AMBER generative |
| Qwen3VL-4B-Instruct Base | sr=0.75 | CHAIR, POPE Overall + per-split, AMBER generative |
| Qwen3VL-4B-Instruct Base | sr=1.00 | CHAIR, POPE Overall + per-split, AMBER generative |

可选：若希望 native control 也证明通用能力不掉，再补 4B base 在各个 sr 下的 MMStar/CV-Bench。

另外，如果想把 blank teacher 的结论扩展到 8B 表格，需要补 8B 的 `tr=0` blank-teacher 实测数值；目前正文只用 2B 的 blank-teacher 行作为极端控制。

AMBER 当前正文只报告 generative CHAIR/HAL。若附录要完整 AMBER，则需要补齐或整理 Cover/COG 以及 discriminative 子指标；这不是主表缺失，而是 appendix 完整性问题。

## P3：如果后续 appendix 要放完整 checkpoint sweep

`instruct_tr025_eval_results.md` 和 `instruct_tr05_eval_results.md` 对 2B/4B/8B 的 RKL/RiskMask 结果都只有 POPE per-split accuracy，没有 POPE Overall Accuracy/F1。当前正文只需要 P0 的四个 selected Res-OPD rows；若 appendix 要完整 sweep，则需要补全：

| 文件 | 范围 | 缺少项 |
|---|---|---|
| `instruct_tr025_eval_results.md` | 2B/4B/8B, RKL + RiskMask, all reported steps | POPE Overall Accuracy/Recall/F1 |
| `instruct_tr05_eval_results.md` | 2B/4B/8B, RKL + RiskMask, all reported steps | POPE Overall Accuracy/Recall/F1 |

## P4：目前正文没用，但 plan 里标注过不完整

`2b_instruct_tr10_sr_ablation_results.md` 顶部说明中提到：

| 范围 | 问题 | 当前正文是否使用 |
|---|---|---|
| sr=0.25 AMBER discriminative | 结果为 0.0，疑似 eval issue | 未使用 |
| sr=0.50/0.75 step 156 | AMBER/MMStar/CV-Bench 不完整 | 未使用，正文只用 step 100 |

这些不是当前正文阻塞项，但如果后面要把 student-side degradation 的 full sweep 放附录，需要一起补齐。

## 当前已经确认可用

- Native Base 2B/8B 在 sr=0.25/0.50/0.75/1.00 下的 CHAIR、POPE Overall F1、AMBER generative：来自 `base_model_eval_results.md`。
- Student-side degradation RKL control 的 step 100 POPE F1：来自 `2b_instruct_tr10_sr_ablation_results.md`。
- 主表 tr=0.75 代表 checkpoint 的 POPE F1：来自 `instruct_masking_ablation_results.md`、`8b_riskmask_top_p_ablation_results.md`、`instruct_sft_ablation_results.md` 中带 Overall F1 的表。
- Probe/Analysis 可用数据：`probes/results/same_image_kl_probe_results/*_summary.{md,json}`、`probes/results/full5k_base_vs_variants_v2/caption_object_decomposition_summary.{md,json}`、`probes/results/conservative_veto_gate_sweep/.../gate_sweep_summary.json`。

# 实验四（Result4 `main_lite`）结果分析摘要

## 1. 结论先行

实验四 `main_lite` 的结果已经基本完成了本轮“单 seed 主实验筛选”的目标，但还没有达到“最终结论定稿”的程度。

本轮最明确的结论有三点：

1. **VMD 仍然是当前最有效的性能提升方向。**
   在本轮基线 `e4_ccg_xlstm_base` 的 OOD RMSE 为 `0.185610` 的前提下，`e4_vmd_ccg_l005_lr3e4_nomix` 将 OOD RMSE 降到 `0.180483`，相对基线下降约 `2.76%`，是本轮 `main_lite` 中 OOD 表现最好的配置。
2. **Phys 在本轮只表现出弱收益，尚不足以单独支撑强结论。**
   最好的 Phys 配置 `e4_ccg_phys_s0010_r0050` 的 OOD RMSE 为 `0.185139`，相对基线仅下降约 `0.25%`。这一改进方向正确，但幅度较小，单 seed 下很难排除随机波动影响。
3. **Joint 说明“VMD + 轻量 physics”有潜力，但当前还没有超过最佳纯 VMD。**
   最优 Joint 配置 `e4_joint_l005_lr5e4_nomix_s0010_r0050` 的 OOD RMSE 为 `0.181138`，优于同主干假设下的 `e4_vmd_ccg_l005_lr5e4_nomix` (`0.181281`)，但仍不如本轮最优纯 VMD `e4_vmd_ccg_l005_lr3e4_nomix` (`0.180483`)。

一句话概括：**实验四 `main_lite` 已经把主线收敛到了 “VMD-no mixer + 更小学习率” 这条方向；Phys 可保留为辅助候选，但当前证据强度明显弱于 VMD。**

---

## 2. 实验目的与结果来源

本次摘要基于以下结果文件：

- 汇总表：`results/result4/outputs/summary/ablation_routine_test.csv`
- 汇总表：`results/result4/outputs/summary/ablation_ood_test.csv`
- 物理指标表：`results/result4/outputs/summary/physics_metrics.csv`
- 实验设计说明：`docs/experiment4_manual.md`

`main_lite` 的设计目的不是直接给出最终论文主表，而是先用一组精简的单 seed 实验回答四个问题：

1. Phys 最合适的权重组合是什么；
2. Phys 的收益更偏向 smooth、roll，还是二者组合；
3. VMD 的收益主要来自更小学习率，还是去掉 state mixer；
4. 在确定单模块 winner 之后，Phys 与 VMD 联合能否继续提升。

因此，下文的判断以“筛选方向”为主，而不是把当前单次结果直接当作正式统计结论。

---

## 3. Baseline 与分组整体表现

本轮统一基线 `e4_ccg_xlstm_base` 的主要结果为：

- `val RMSE = 0.030417`
- `routine_test RMSE = 0.031318`
- `ood_test RMSE = 0.185610`

以该基线为参考，本轮三类增强方向的整体表现可以概括为：

- **Phys 组**：分布内和 OOD 都只有轻微变化，整体提升幅度较小；
- **VMD 组**：在 `val`、`routine_test`、`ood_test` 三个 split 上都明显优于基线，是本轮最强增强方向；
- **Joint 组**：整体优于基线，也优于部分 VMD 候选，但暂时还没有击败本轮最佳纯 VMD。

从排序结果看：

- `val` 最优：`e4_vmd_ccg_l003_lr5e4_nomix = 0.024470`
- `routine_test` 最优：`e4_vmd_ccg_l003_lr5e4_nomix = 0.024643`
- `ood_test` 最优：`e4_vmd_ccg_l005_lr3e4_nomix = 0.180483`

这说明当前已经出现了一个很重要的信号：**最佳验证误差配置与最佳 OOD 配置并不完全一致。**

---

## 4. Phys 组分析

本轮 `main_lite` 中 Phys 组的 OOD RMSE 为：

- `e4_ccg_phys_s0010_r0000`：`0.185143`
- `e4_ccg_phys_s0000_r0025`：`0.185215`
- `e4_ccg_phys_s0000_r0050`：`0.186583`
- `e4_ccg_phys_s0010_r0050`：`0.185139`
- `e4_ccg_phys_s0010_r0075`：`0.185159`

可以得到以下判断：

1. **Phys 的最优候选仍然集中在轻量约束区间。**
   `smooth=0.0010, roll=0.0050` 与 `smooth=0.0010, roll=0.0075` 都比基线略好，说明实验设计里“轻量 regularization”这个判断仍然成立。
2. **单独 smooth、单独轻量 roll、以及 smooth+roll 的 OOD 差异非常小。**
   `0.185143 / 0.185215 / 0.185139` 之间的差距非常有限，因此当前不能得出“收益主要来自某一个物理项”的强结论。
3. **较强 roll-only 配置不稳。**
   `e4_ccg_phys_s0000_r0050` 的 `val RMSE = 0.029683`、`routine_test RMSE = 0.030083`，都优于基线，但 `ood_test RMSE = 0.186583` 反而差于基线。这说明稍强的 roll-only 更像是在改善分布内拟合，而不是改善 OOD 泛化。

如果只从本轮单 seed 主实验给出工作结论，Phys 组建议保留两个候选：

- **误差折中候选**：`e4_ccg_phys_s0010_r0050`
- **物理指标候选**：`e4_ccg_phys_s0010_r0075`

其中后者在 `physics_metrics.csv` 中的 OOD `smoothness=6.50e-06`、`roll_consistency_rmse=0.001774` 更好，说明如果论文后续需要讨论“预测误差与物理一致性的折中”，它比 `r0050` 更适合作为补充证据。

---

## 5. VMD 组分析

本轮 VMD 组的结论最清晰。

主要配置的 OOD RMSE 为：

- `e4_vmd_ccg_l005_lr5e4_mix`：`0.182976`
- `e4_vmd_ccg_l005_lr5e4_nomix`：`0.181281`
- `e4_vmd_ccg_l005_lr3e4_nomix`：`0.180483`
- `e4_vmd_ccg_l005_lr7e4_nomix`：`0.181878`
- `e4_vmd_ccg_l003_lr5e4_nomix`：`0.181643`
- `e4_vmd_ccg_l008_lr5e4_nomix`：`0.182046`

可以得到三个明确判断：

1. **去掉 state mixer 是有效的。**
   在相同 `lambda_vmd=0.05, lr=5e-4` 条件下，`mix -> nomix` 使 OOD RMSE 从 `0.182976` 降到 `0.181281`。这说明 `state mixer` 在当前 VMD 结构下更可能引入额外耦合，而不是提供稳定增益。
2. **更小学习率继续改善 OOD。**
   在 `nomix` 条件下，`lr=3e-4` 优于 `5e-4`，而 `7e-4` 又明显回退，说明本轮 OOD 最优区间更偏向较小学习率。
3. **最佳 `val/routine` 与最佳 OOD 已经分离。**
   `e4_vmd_ccg_l003_lr5e4_nomix` 拿到了本轮最优 `val` 和 `routine_test`，但不是最优 OOD；真正的 OOD winner 是 `e4_vmd_ccg_l005_lr3e4_nomix`。这意味着后续如果仍然只按 `val_rmse_mean` 选模，会与 OOD 优化目标发生偏移。

因此，本轮最值得保留的 VMD 候选应当是：

- **OOD 主候选**：`e4_vmd_ccg_l005_lr3e4_nomix`
- **分布内/验证主候选**：`e4_vmd_ccg_l003_lr5e4_nomix`
- **结构消融参考**：`e4_vmd_ccg_l005_lr5e4_mix`

其中第一项应视为当前实验四主线。

---

## 6. Joint 组分析

Joint 组当前用于验证的问题是：在固定 `VMD-no mixer, lr=5e-4, lambda_vmd=0.05` 的主干假设下，叠加轻量 physics 是否还有增益。

主要结果为：

- `e4_joint_l005_lr5e4_nomix_r0025`：`ood RMSE = 0.181529`
- `e4_joint_l005_lr5e4_nomix_r0050`：`ood RMSE = 0.181485`
- `e4_joint_l005_lr5e4_nomix_s0010_r0050`：`ood RMSE = 0.181138`
- `e4_joint_l005_lr5e4_nomix_s0010_r0075`：`ood RMSE = 0.181746`

可以得到两个结论：

1. **在 `l005_lr5e4_nomix` 这条 VMD 主干上，叠加轻量 physics 是有小幅收益的。**
   `e4_joint_l005_lr5e4_nomix_s0010_r0050 = 0.181138` 略优于 `e4_vmd_ccg_l005_lr5e4_nomix = 0.181281`。
2. **但当前 Joint 仍未超过本轮最优纯 VMD。**
   本轮最好的 OOD 配置依然是 `e4_vmd_ccg_l005_lr3e4_nomix = 0.180483`。因此，当前还不能得出“联合模型优于最佳单模块”的结论。

不过，Joint 在物理指标上的表现值得保留。相较 Phys-only 候选，Joint 的 OOD `roll_consistency_rmse` 已下降到 `0.00137` 左右，优于本轮 Phys-only 的 `0.00177~0.00187` 区间。这说明 **physics 约束在 VMD 表征上更容易转化为可观测的一致性改善**。

这也意味着下一步真正值得补的不是更多 `lr=5e-4` Joint，而是：

- 继承本轮最优 VMD 主干 `l005_lr3e4_nomix`
- 再叠加 `r0025/r0050/s0010_r0050`

否则当前 Joint 结论的说服力会被“主干本身不是最优”这个问题削弱。

---

## 7. 与前一轮结果的关系

和 `result3` 对照时，需要特别注意：**本轮 baseline 绝对值比上一轮更差，因此跨轮比较更应关注相对增益，而不是只看绝对最优数值。**

已知对照如下：

- `result3` baseline OOD RMSE：`0.182219`
- `result4` baseline OOD RMSE：`0.185610`

在这个前提下：

1. **Phys 没有复现上一轮那种明显的 OOD 改善。**
   `result3` 中 Phys 最优达到 `0.180006`，而本轮 Phys 最优只有 `0.185139`。因此，本轮不应把 Phys 作为最强主线。
2. **VMD 的主线更加清晰。**
   `result3` 中较好的 VMD OOD 结果约为 `0.181180`，本轮进一步到 `0.180483`。这说明 `nomix + 更小 lr` 这条思路在跨轮上是连续成立的。
3. **Joint 的作用更像“在指定 VMD 主干上做细修”，而不是已经独立胜出。**

因此，从跨轮角度看，实验四并不是把实验三完全推翻，而是把其中“VMD 需要改结构和学习率”这条线进一步做实了。

---

## 8. 是否达到实验四 `main_lite` 的阶段目标

| 子目标 | 判断 | 依据 |
|---|---|---|
| 筛出本轮统一 baseline | 达到 | `e4_ccg_xlstm_base` 已完成本轮统一口径。 |
| 判断 Phys 是否值得继续保留 | 达到 | Phys 有轻微正向信号，但证据强度偏弱。 |
| 判断 VMD 收益来源 | 达到 | `nomix` 与小学习率的增益趋势清晰。 |
| 判断 Joint 是否具备继续价值 | 达到 | Joint 有小幅正向信号，但尚未超过最佳纯 VMD。 |
| 支撑正式论文定稿 | 暂不充分 | 还缺多 seed、补充 Joint 主干验证，以及更细的选模分析。 |

总体上，**实验四 `main_lite` 达到了“筛选主线”的目的，但还没有达到“正式定稿”的目的。**

---

## 9. 后续建议

1. **优先进入多 seed 的候选应收缩到 4~6 个，而不是继续扩散。**
   建议至少保留：
   - `e4_ccg_xlstm_base`
   - `e4_ccg_phys_s0010_r0050`
   - `e4_ccg_phys_s0010_r0075`
   - `e4_vmd_ccg_l005_lr3e4_nomix`
   - `e4_vmd_ccg_l003_lr5e4_nomix`
   - `e4_joint_l005_lr5e4_nomix_s0010_r0050`
2. **如果只能补少量新实验，优先补 Joint 继承最优 VMD 主干。**
   即新增 `l005_lr3e4_nomix + physics`，而不是继续补更多 `lr=5e-4` Joint。
3. **后续分析不应只依赖 `val_rmse_mean`。**
   当前已经出现“最佳 val 不等于最佳 OOD”的情况，建议在 `diag` 或后续总结中补充：
   - `val_pred_loss`
   - `val_vmd_aux_loss`
   - `val_phys_loss`
   - `smoothness`
   - `roll_consistency_rmse`
4. **Phys 的结论必须依赖多 seed 才能确认。**
   目前 Phys 的提升幅度过小，单 seed 下不适合写成强结论。

---

## 10. 可直接写入论文/实验记录的简短表述

实验四的 `main_lite` 单 seed 主实验表明，VMD 模块仍然是当前最有效的性能增强方向，其中去除 state mixer 并适当降低学习率可显著改善常规测试集与 OOD 测试集表现，最佳配置 `e4_vmd_ccg_l005_lr3e4_nomix` 在 OOD 测试集上取得了本轮最低 RMSE。相比之下，物理约束在本轮仅表现出幅度较小的正向收益，说明其对泛化的帮助可能存在，但稳定性和统计显著性仍需多随机种子进一步确认。联合模型在固定的 VMD 主干上较纯 VMD 略有改善，并在 roll consistency 等物理一致性指标上表现更优，说明物理约束与 VMD 表征之间存在一定互补性；但由于当前联合组尚未超过本轮最佳纯 VMD 配置，因此实验四的后续重点应放在“以最优 VMD 主干为基础重新构造 Joint 候选”，并通过多 seed 复验筛选最终结论。

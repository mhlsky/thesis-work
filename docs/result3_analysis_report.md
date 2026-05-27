# 实验三（Result3）结果分析报告

## 1. 结论先行

实验三的结果**基本达到了“独立验证 Phys 与 VMD 作用”的实验目的**：

1. **Phys 组**证明了物理约束确实可能提升 OOD 泛化：`CCG-Phys (smooth+roll, conservative)` 在 OOD test 上取得全组最低 RMSE=0.180006，相对 CCG Baseline 下降 1.21%。  
2. **VMD 组**证明了 VMD 对常规场景有明显收益，但对 OOD 的收益依赖训练超参和结构：`VMD-CCG (no state mixer)` 在 routine test 上最优，RMSE=0.026007，相对基线下降 16.01%；`VMD-CCG (lr=5e-4)` 是 VMD 组中 OOD 预测最稳的方案，OOD RMSE 下降 0.57%。
3. 但实验三也暴露出一个关键问题：**预测误差最优的 Phys 配置并不等于物理一致性最优**。例如 conservative 配置 OOD RMSE 最优，但按统一方式重算的 smoothness 和 roll consistency 均劣于基线；smooth-only 的物理指标更好，但 OOD 误差变差。因此，物理约束目前更适合作为“可调的正则项”，还不能直接宣称联合物理指标稳定提升。

一句话总结：**实验三完成了筛选和定位问题的目的，但还不宜直接把当前配置作为最终模型结论；后续应围绕 Phys 权重/roll loss 形式和 VMD-state mixer 交互继续微调。**

---

## 2. 实验目的与结果来源

本次结果来自：

- 汇总表：`results/result3/outputs/summary/ablation_routine_test.csv`
- 汇总表：`results/result3/outputs/summary/ablation_ood_test.csv`
- 官方物理指标表：`results/result3/outputs/summary/physics_metrics.csv`
- 本报告补充生成的可比物理指标：`results/result3/outputs/summary/comparable_physics_metrics.csv`
- 实验清单：`configs/ablation_result3.yaml`
- 批量脚本：`scripts/run_result3_experiments.sh`

实验三的设计目的不是继续把 Phys 和 VMD 耦合在一起，而是分成两组独立判断：

- **Phys 独立验证组**：在 `CCG-xLSTM` 基线上分别加入 smooth、roll、smooth+roll，判断物理损失是否改善 OOD 或物理指标。
- **VMD 独立验证组**：在 `VMD-CCG` 上改变 warmup、学习率、VMD 分解参数、是否启用 state mixer，判断 VMD 的有效性与不稳定来源。

说明：官方 `physics_metrics.csv` 只包含 `physics.enabled=true` 的 Phys 组。为了公平比较 baseline 和 VMD，本报告使用保存的 `predictions_*.npz` 按同一公式补充重算了所有模型的 smoothness(p/r/phi) 和 trapezoid roll consistency。两个指标均为**越小越好**。

---

## 3. Routine test 预测性能

| 模型 | MAE | RMSE | RMSE相对基线 | R² |
|---|---:|---:|---:|---:|
| CCG Baseline | 0.014189 | 0.030964 | +0.00% | 0.999834439 |
| CCG-Phys (smooth only) | 0.014277 | 0.031577 | +1.98% | 0.999827824 |
| CCG-Phys (roll only) | 0.014026 | 0.031974 | +3.26% | 0.999823463 |
| CCG-Phys (smooth+roll, conservative) | 0.013532 | 0.030882 | -0.27% | 0.999835317 |
| CCG-Phys (smooth+roll, standard) | 0.013509 | 0.031011 | +0.15% | 0.999833934 |
| VMD-CCG (l=0.05 base) | 0.013291 | 0.029480 | -4.79% | 0.999849926 |
| VMD-CCG (warmup=10) | 0.014032 | 0.031220 | +0.83% | 0.999831694 |
| VMD-CCG (lr=5e-4) | 0.012351 | 0.027519 | -11.13% | 0.999869235 |
| VMD-CCG (K=2, alpha=1000) | 0.015144 | 0.034237 | +10.57% | 0.999797596 |
| VMD-CCG (K=4, alpha=3000) | 0.013231 | 0.031077 | +0.37% | 0.999833228 |
| VMD-CCG (no state mixer) | 0.011853 | 0.026007 | -16.01% | 0.999883210 |

### 观察

- routine test 最好的是 `VMD-CCG (no state mixer)`，RMSE 相对基线下降 16.01%，说明 **VMD 在常规分布内有明显帮助**。
- `VMD-CCG (lr=5e-4)` 也明显优于基线，RMSE 下降 11.13%。
- Phys 组整体提升较小：conservative 配置 RMSE 仅下降 0.27%，但 MAE 下降更明显，说明 Phys 对平均绝对误差有一定稳定作用。
- `K=2, alpha=1000` 是明显负向配置，routine RMSE 比基线上升 10.57%。

---

## 4. OOD test 预测性能

| 模型 | MAE | RMSE | RMSE相对基线 | R² |
|---|---:|---:|---:|---:|
| CCG Baseline | 0.086354 | 0.182219 | +0.00% | 0.997219419 |
| CCG-Phys (smooth only) | 0.088277 | 0.185912 | +2.03% | 0.997105573 |
| CCG-Phys (roll only) | 0.085648 | 0.180349 | -1.03% | 0.997276197 |
| CCG-Phys (smooth+roll, conservative) | 0.085528 | 0.180006 | -1.21% | 0.997286535 |
| CCG-Phys (smooth+roll, standard) | 0.089068 | 0.188969 | +3.70% | 0.997009610 |
| VMD-CCG (l=0.05 base) | 0.089009 | 0.187383 | +2.83% | 0.997059602 |
| VMD-CCG (warmup=10) | 0.088343 | 0.185969 | +2.06% | 0.997103796 |
| VMD-CCG (lr=5e-4) | 0.085659 | 0.181180 | -0.57% | 0.997251046 |
| VMD-CCG (K=2, alpha=1000) | 0.091903 | 0.192455 | +5.62% | 0.996898252 |
| VMD-CCG (K=4, alpha=3000) | 0.089413 | 0.189075 | +3.76% | 0.997006242 |
| VMD-CCG (no state mixer) | 0.085981 | 0.182272 | +0.03% | 0.997217796 |

### 观察

- OOD test 最好的是 `CCG-Phys (smooth+roll, conservative)`，RMSE 相对基线下降 1.21%，说明**低权重 smooth+roll 组合确实能提升跨分布预测**。
- `CCG-Phys (roll only)` 也有 OOD RMSE 改善，下降 1.03%。
- `VMD-CCG (lr=5e-4)` 是 VMD 组中 OOD 最优，RMSE 下降 0.57%，幅度不大但方向正确。
- `VMD-CCG (l=0.05 base)`、`warmup=10`、`K=4`、`K=2` 在 OOD 上均劣于基线，说明 **VMD 本身不是无条件提升，必须配合合适学习率/结构**。
- `VMD-CCG (no state mixer)` 的 OOD RMSE 基本与基线持平（+0.03%），但 routine 大幅提升，说明去掉 state mixer 后模型更适合常规测试集，对 OOD 的帮助有限。

---

## 5. 关键状态量 RMSE 对比

### 5.1 Routine test

| 模型 | u RMSE | v RMSE | p RMSE | r RMSE | phi RMSE |
|---|---:|---:|---:|---:|---:|
| CCG Baseline | 0.063592 (+0.00%) | 0.026413 (+0.00%) | 0.004045 (+0.00%) | 0.002148 (+0.00%) | 0.005595 (+0.00%) |
| CCG-Phys (smooth+roll, conservative) | 0.064275 (+1.07%) | 0.024178 (-8.46%) | 0.004056 (+0.26%) | 0.002134 (-0.66%) | 0.005622 (+0.48%) |
| VMD-CCG (lr=5e-4) | 0.056207 (-11.61%) | 0.023970 (-9.25%) | 0.004044 (-0.02%) | 0.002098 (-2.34%) | 0.005644 (+0.87%) |
| VMD-CCG (no state mixer) | 0.051887 (-18.41%) | 0.025226 (-4.50%) | 0.004037 (-0.21%) | 0.002157 (+0.40%) | 0.005674 (+1.41%) |

### 5.2 OOD test

| 模型 | u RMSE | v RMSE | p RMSE | r RMSE | phi RMSE |
|---|---:|---:|---:|---:|---:|
| CCG Baseline | 0.314559 (+0.00%) | 0.258192 (+0.00%) | 0.006423 (+0.00%) | 0.008827 (+0.00%) | 0.016997 (+0.00%) |
| CCG-Phys (smooth+roll, conservative) | 0.309283 (-1.68%) | 0.256779 (-0.55%) | 0.006400 (-0.36%) | 0.008793 (-0.38%) | 0.017373 (+2.21%) |
| VMD-CCG (lr=5e-4) | 0.312655 (-0.61%) | 0.256861 (-0.52%) | 0.006367 (-0.88%) | 0.008745 (-0.93%) | 0.016803 (-1.14%) |
| VMD-CCG (no state mixer) | 0.310048 (-1.43%) | 0.263785 (+2.17%) | 0.006401 (-0.35%) | 0.008518 (-3.50%) | 0.017018 (+0.12%) |

### 观察

- routine 的主要收益来自 `u` 和 `v`：`VMD-CCG (no state mixer)` 的 `u` RMSE 下降 18.41%，`VMD-CCG (lr=5e-4)` 的 `u` RMSE 下降 11.61%。
- OOD 上 `VMD-CCG (lr=5e-4)` 的五个状态量均有小幅改善，是 VMD 组里最均衡的配置。
- `CCG-Phys (smooth+roll, conservative)` 的 OOD 改善主要来自 `u/v/p/r`，但 `phi` RMSE 上升 2.21%，提示 roll/phi 相关约束仍需进一步检查。

---

## 6. 物理一致性指标

### 6.1 Phys 组 OOD 可比物理指标

| 模型 | smoothness(p/r/phi) | 相对基线 | roll_consistency_rmse | 相对基线 |
|---|---:|---:|---:|---:|
| CCG Baseline | 8.189e-06 | +0.00% | 1.354e-03 | +0.00% |
| CCG-Phys (smooth only) | 7.387e-06 | -9.80% | 1.324e-03 | -2.25% |
| CCG-Phys (roll only) | 8.954e-06 | +9.35% | 1.642e-03 | +21.30% |
| CCG-Phys (smooth+roll, conservative) | 8.953e-06 | +9.33% | 1.784e-03 | +31.73% |
| CCG-Phys (smooth+roll, standard) | 6.964e-06 | -14.96% | 1.588e-03 | +17.28% |

### 6.2 VMD 组 OOD 可比物理指标

| 模型 | smoothness(p/r/phi) | 相对基线 | roll_consistency_rmse | 相对基线 |
|---|---:|---:|---:|---:|
| CCG Baseline | 8.189e-06 | +0.00% | 1.354e-03 | +0.00% |
| VMD-CCG (l=0.05 base) | 6.772e-06 | -17.30% | 2.067e-03 | +52.64% |
| VMD-CCG (warmup=10) | 7.982e-06 | -2.53% | 1.849e-03 | +36.53% |
| VMD-CCG (lr=5e-4) | 8.337e-06 | +1.81% | 1.889e-03 | +39.49% |
| VMD-CCG (K=2, alpha=1000) | 1.036e-05 | +26.56% | 2.103e-03 | +55.32% |
| VMD-CCG (K=4, alpha=3000) | 8.446e-06 | +3.14% | 1.884e-03 | +39.17% |
| VMD-CCG (no state mixer) | 8.286e-06 | +1.19% | 1.120e-03 | -17.27% |

### 观察

- Phys 组中，`smooth only` 的物理指标最好：smoothness 下降 9.80%，roll consistency 下降 2.25%；但它的 OOD RMSE 比基线上升 2.03%。
- `smooth+roll, conservative` 虽然 OOD RMSE 最好，但 smoothness 与 roll consistency 都差于基线，说明该配置的 OOD 提升主要来自预测误差层面，不能同时证明物理一致性提升。
- VMD 组中，`no state mixer` 的 roll consistency 最好，OOD roll consistency 相对基线下降 17.27%，这支持“state mixer 与 VMD 后处理可能引入不稳定耦合”的判断。
- `VMD-CCG (l=0.05 base)` 的 OOD smoothness 改善，但 roll consistency 变差较多，说明 VMD 分量学习可能让预测曲线更平滑，但不自动保证 `p-phi` 积分一致。

---

## 7. 训练过程风险检查

| 模型 | 记录epoch数 | 最优epoch | 最优val_loss | 最后val_loss |
|---|---:|---:|---:|---:|
| CCG Baseline | 16 | 6 | 0.103006 | 0.127921 |
| CCG-Phys (smooth only) | 16 | 6 | 0.102936 | 0.126311 |
| CCG-Phys (roll only) | 15 | 6 | 0.102916 | 0.133953 |
| CCG-Phys (smooth+roll, conservative) | 15 | 6 | 0.103110 | 0.126521 |
| CCG-Phys (smooth+roll, standard) | 18 | 6 | 0.102719 | 0.137517 |
| VMD-CCG (l=0.05 base) | 18 | 2 | 0.116207 | 0.161022 |
| VMD-CCG (warmup=10) | 15 | 2 | 0.112345 | 0.143341 |
| VMD-CCG (lr=5e-4) | 17 | 2 | 0.116461 | 0.143581 |
| VMD-CCG (K=2, alpha=1000) | 10 | 1 | 0.119406 | 0.122896 |
| VMD-CCG (K=4, alpha=3000) | 20 | 2 | 0.114626 | 0.162746 |
| VMD-CCG (no state mixer) | 16 | 2 | 0.116343 | 0.140140 |

### 观察

- Phys 组最优验证损失基本出现在第 6 个 epoch 左右，后续验证损失升高，说明早停保存 best checkpoint 是必要的。
- VMD 组最优验证损失多出现在第 1-2 个 epoch，后续验证损失明显升高，说明 VMD 相关配置更容易过拟合或训练后期漂移。
- 因此，VMD 组后续如果用于正式结论，建议重点检查学习率、warmup、早停 patience、`lambda_vmd` 的组合，而不是只比较最终 epoch。

---

## 8. 是否达到实验目的

| 子目标 | 判断 | 依据 |
|---|---|---|
| 分离 Phys 与 VMD，避免耦合解释 | 达到 | result3 脚本与 ablation 清单已将 Phys/VMD 拆成独立组。 |
| 验证 Phys 是否能改善 OOD 或物理指标 | 部分达到 | conservative 配置 OOD RMSE 最优；smooth-only 物理指标改善，但二者没有同时达到。 |
| 验证 VMD 是否有效 | 部分达到 | routine 提升明显；OOD 只有 lr=5e-4 小幅改善，默认 VMD 和部分 K/alpha 配置反而变差。 |
| 定位 VMD 不稳定来源 | 达到 | no state mixer 在 routine 与 roll consistency 上表现突出，提示 state mixer 与 VMD 交互需谨慎。 |
| 支撑最终模型定稿 | 暂不充分 | 还缺少多随机种子、图表、最终 Phys+VMD 联合配置的重新筛选。 |

总体结论：**实验三达到了“诊断与筛选”的目的，但还没有完全达到“最终模型定稿”的目的。**

---

## 9. 后续建议

1. **若论文优先强调 OOD 预测性能**：优先保留 `CCG-Phys (smooth+roll, conservative)` 作为 Phys 方向证据，并说明其 OOD RMSE 最优。
2. **若论文优先强调物理一致性**：优先讨论 `CCG-Phys (smooth only)`，因为它在 OOD smoothness 与 roll consistency 上都优于基线，但需要承认误差有牺牲。
3. **VMD 后续推荐配置**：以 `VMD-CCG (lr=5e-4)` 作为 OOD 性能候选，以 `VMD-CCG (no state mixer)` 作为结构消融证据。
4. **不建议直接使用 `K=2, alpha=1000`**：该配置在 routine 和 OOD 上均明显变差。
5. **需要补充稳健性验证**：至少对最关键的 3 个配置（baseline、Phys conservative、VMD lr=5e-4/no mixer）补 2-3 个随机种子，避免单次训练偶然性影响论文结论。
6. **需要重新审视 roll loss**：当前 roll 相关损失没有稳定降低 roll consistency，建议检查 loss 与评估指标是否完全一致、`p/phi` 单位是否匹配、dt 是否合理，以及 roll 权重是否过大。

---

## 10. 可直接写入论文的简短表述

实验三将物理约束和 VMD 分解模块拆分为两组独立消融。结果显示，低权重物理约束在 OOD 测试集上取得最佳整体预测误差，说明物理先验有助于提升跨分布泛化；但不同物理损失之间存在权衡，smooth-only 更有利于物理一致性，而 smooth+roll conservative 更有利于 OOD 误差。VMD 模块在常规测试集上带来显著误差下降，但其 OOD 效果对学习率、分解参数和状态耦合结构较敏感，其中 lr=5e-4 是较稳健的 OOD 配置，而 no state mixer 在 routine 与 roll consistency 上表现最好。综合来看，实验三验证了 Phys 与 VMD 的潜在价值，也指出最终模型需要在误差和物理一致性之间进一步做权衡优化。


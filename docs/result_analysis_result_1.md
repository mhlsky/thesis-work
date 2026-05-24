# result_1 结果分析与改进建议

> 分析时间：2026-05-24  
> 分析对象：`results/result_1/logs/`、`results/result_1/outputs/`

## 1. 先说结论

- **大部分训练与评估任务已经正常跑完**，各主线模型目录下都能看到：
  - `best.pt`
  - `metrics_val.json`
  - `metrics_routine_test.json`
  - `metrics_ood_test.json`
  - `predictions_*.npz`
  - `train_log.csv`
- **最明显的流程问题**出现在 `results/result_1/logs/summarize_results.log`：汇总脚本因为缺少 `persistence_seq128_pred10` 目录而直接报错退出，导致 `outputs/summary/` 原先没有成功生成汇总表。
- **主实验结果没有完全达到 Step 07 的预期效果**：
  - `CCG-xLSTM` 在 **OOD** 上表现最好，说明“条件门控”是有效的；
  - 但 `VMD-CCG-xLSTM` 和 `VMD-CCG-Phys-xLSTM` 的误差明显变差，说明 **VMD 辅助监督 + 物理约束这一条主线目前没有验证成功**；
  - 最终模型 `VMD-CCG-Phys-xLSTM` **没有在误差指标上优于前一阶段模型**，因此目前还不能作为“最终最优主模型”来写论文主结论。

---

## 2. 运行检查结果

## 2.1 日志总体情况

- `train_*.log`、`eval_*.log` 未发现 `Traceback`、`NaN`、`OOM` 等明显异常。
- `step03_build_vmd_cache.log` 显示 VMD 缓存构建完成，数量如下：
  - train: 57
  - validation: 9
  - routine_test: 30
  - ood_test: 29
- 说明 **训练/评估主链路本身基本是通的**。

## 2.2 汇总脚本异常

`results/result_1/logs/summarize_results.log` 报错：

- 缺失目录：`outputs/persistence_seq128_pred10`
- 直接触发：
  - `FileNotFoundError: Run directory not found: outputs/persistence_seq128_pred10`

这说明当前 `configs/ablation_list.yaml` 中把 `Persistence` 作为必汇总项，但这次服务器结果里 **没有对应正式输出目录**。

## 2.3 已做修复

我已修改：

- `src/ship_motion/summarize_results.py`

修复内容：

1. 当某个 run 目录缺失时，**默认跳过并给出警告**，而不是整次汇总直接崩溃；
2. 当某个 run 缺失 `metrics_routine_test.json` 或 `metrics_ood_test.json` 时，也会跳过并提示；
3. 新增 `--strict-missing-runs` 开关；如果你后面想强制检查完整性，仍然可以恢复“缺失即失败”的严格模式。

修复后，我已重新在本地对这批结果执行汇总，已生成：

- `results/result_1/outputs/summary/ablation_routine_test.csv`
- `results/result_1/outputs/summary/ablation_ood_test.csv`
- `results/result_1/outputs/summary/physics_metrics.csv`

---

## 3. 实验结果分析

## 3.1 Routine Test（常规测试集）

按 `rmse_mean` 看：

| 排名 | 模型 | RMSE_mean | 结论 |
|---|---|---:|---|
| 1 | Transformer | 0.030851 | 常规集最优 |
| 2 | CCG-xLSTM | 0.030964 | 非常接近最优 |
| 3 | Lite-xLSTM | 0.030994 | 与前两者差距极小 |
| 4 | GRU | 0.032244 | 稳定基线 |
| 5 | LSTM | 0.036784 | 最弱循环基线 |
| 6 | VMD-CCG-Phys-xLSTM | 0.046727 | 明显退化 |
| 7 | VMD-CCG-xLSTM | 0.048171 | 明显退化 |

关键观察：

- `CCG-xLSTM` 相比 `Transformer` **并没有在常规集上拉开优势**，两者几乎打平；
- 一旦加入 VMD，误差大幅上升：
  - `VMD-CCG-xLSTM` 相比 `CCG-xLSTM`，`rmse_mean` **上升约 55.6%**；
  - `VMD-CCG-Phys-xLSTM` 相比 `VMD-CCG-xLSTM` 虽有小幅回升，但仍远差于不带 VMD 的模型。

## 3.2 OOD Test（分布外测试集）

按 `rmse_mean` 看：

| 排名 | 模型 | RMSE_mean | 结论 |
|---|---|---:|---|
| 1 | CCG-xLSTM | 0.182219 | OOD 最优 |
| 2 | Transformer | 0.185287 | 次优 |
| 3 | GRU | 0.188521 | 也比较稳 |
| 4 | Lite-xLSTM | 0.203533 | 不如 CCG/Transformer |
| 5 | LSTM | 0.206998 | 较弱 |
| 6 | VMD-CCG-xLSTM | 0.234516 | 明显退化 |
| 7 | VMD-CCG-Phys-xLSTM | 0.235505 | 明显退化 |

关键观察：

- `CCG-xLSTM` 相比 `Transformer` 在 OOD 上 **下降约 1.66% 的 RMSE**，这是目前最有价值的正结果；
- 但加入 VMD 后再次显著变差：
  - `VMD-CCG-xLSTM` 相比 `CCG-xLSTM`，`rmse_mean` **上升约 28.7%**；
  - `VMD-CCG-Phys-xLSTM` 相比 `VMD-CCG-xLSTM` 在 OOD 上 **没有误差改善**，反而略差。

## 3.3 分变量结果

从 OOD 指标看，最终两条退化最严重的变量主要还是：

- `u`
- `v`

例如：

- `CCG-xLSTM` 的 `OOD rmse_u = 0.3146`
- `VMD-CCG-Phys-xLSTM` 的 `OOD rmse_u = 0.4452`

这说明当前 VMD/物理约束方案对主运动状态的拟合产生了明显副作用，而不仅仅是少数边缘变量抖动。

---

## 4. 物理一致性分析

虽然 `physics_metrics.csv` 里只有最终模型原始导出值，但我又基于保存下来的 `predictions_*.npz` 补算了 `VMD-CCG-xLSTM` 的物理指标，用于正面对比。

| 模型 | split | smoothness ↓ | roll_consistency_rmse ↓ |
|---|---|---:|---:|
| VMD-CCG-xLSTM | routine_test | 0.000722 | 0.003022 |
| VMD-CCG-Phys-xLSTM | routine_test | 0.000671 | 0.003018 |
| VMD-CCG-xLSTM | ood_test | 0.001843 | 0.004861 |
| VMD-CCG-Phys-xLSTM | ood_test | 0.001971 | 0.004834 |

结论：

- `roll_consistency_rmse` 上，最终模型 **只有非常轻微的改善**；
- `smoothness` 上，最终模型：
  - 在 `routine_test` 更平滑；
  - 但在 `ood_test` 反而略差。

所以目前不能说“虽然误差没提升，但物理一致性显著提升”。  
**当前物理收益存在，但幅度很小，不足以支撑误差明显退化的代价。**

---

## 5. 是否达到预期效果

结合 `docs/implementation_steps/07_experiments.md` 中的目标，当前结论是：

## 5.1 已达到的部分

- 主线大部分模型已经完成训练与评估；
- routine / OOD 指标文件已齐；
- 可以生成消融汇总表；
- 已经能看出一个明确结论：**CCG 条件门控对 OOD 泛化是有效的**。

## 5.2 未达到的部分

- **主线消融不完整**：缺少 `Persistence` 正式结果；
- **最终主模型没有成为最优模型**；
- **VMD 与物理约束没有带来论文预期中的主指标提升**；
- 目前不适合直接写出“最终模型全面优于基线”的结论。

因此，**这批结果只能证明“CCG 模块有效”，不能证明“VMD-CCG-Phys-xLSTM 是最终最优方案”。**

---

## 6. 最可能的问题原因

## 6.1 VMD 辅助损失权重可能偏大

当前配置：

- `lambda_vmd: 0.2`

从结果看，VMD 一加就整体退化，说明很可能出现了：

- 模型被迫过度拟合 VMD 模态重构；
- 辅助任务干扰了主预测任务；
- 特别是在 OOD 上，这种辅助监督没有带来更强泛化，反而引入额外偏差。

## 6.2 物理约束权重偏激进

当前配置：

- `lambda_smooth: 0.01`
- `lambda_roll: 0.05`

而项目文档在调试顺序里明确建议先从更小权重开始：

- `lambda_smooth = 0.001`
- `lambda_roll = 0.01`

所以当前正式权重对现在这组模型来说，**大概率偏大**。

## 6.3 “VMD 主线”本身尚未验证收益

从 `CCG-xLSTM -> VMD-CCG-xLSTM` 这一步就已经大退化，说明问题不只是物理约束，**更早在 Step 05 就已经开始了**。

换句话说：

- 先不要把锅全甩给 physics；
- 应先确认 **VMD 融合策略本身是否有效**。

---

## 7. 下一轮改进建议（按优先级排序）

## P0：先补齐最基础缺口

1. **补跑 Persistence 正式结果**
   - 现在只有 smoke 版本，正式 `result_1` 中缺失；
   - 没有它，主线消融表不完整。
2. 重新生成 summary 和图表；
3. 论文里主表至少先保证 baseline 完整。

## P1：先救 Step 05，再看 Step 06

下一轮实验不要直接继续堆最终模型，建议按下面顺序回退排查：

1. 以 `CCG-xLSTM` 作为当前可靠参照；
2. 重新跑 `VMD-CCG-xLSTM`，把
   - `lambda_vmd: 0.2 -> 0.05`
   - 如果还不行，再试 `0.1`
3. 如果 Step 05 仍然不提升，先暂停 physics 分支；
4. 只有当 Step 05 至少不退化时，再继续 Step 06。

## P1：降低 physics 权重，按文档推荐顺序渐进

建议重新试一组更保守的物理权重：

第一组：

- `lambda_smooth: 0.001`
- `lambda_roll: 0.01`

第二组：

- `lambda_smooth: 0.003`
- `lambda_roll: 0.02`

只有当误差基本不恶化时，再考虑回到更强权重。

## P1：重点盯 OOD 的 `u` / `v`

下一轮调参时，不要只看 `rmse_mean`，一定同步看：

- `rmse_u`
- `rmse_v`

因为这两个变量是当前退化最明显的位置，最容易反映 VMD/physics 是否真的有帮助。

## P2：补一个最关键的对照结论

就当前结果而言，论文最稳的一条线不是“最终模型最好”，而是：

> 在当前实验设置下，CCG 条件门控对分布外泛化最有效；  
> 但 VMD 辅助监督和物理约束的收益尚不稳定，仍需进一步调权重与训练策略。

这个结论是当前数据真正支持的。

## P2：建议增加的最小补充实验

如果算力有限，优先补这 4 个：

1. `persistence_seq128_pred10` 正式评估
2. `vmd_ccg_xlstm` with `lambda_vmd=0.05`
3. `vmd_ccg_phys_xlstm` with `lambda_smooth=0.001, lambda_roll=0.01`
4. `ccg_xlstm` 再复现实验 1 次（确认当前最好 OOD 结果稳定）

这样就能快速回答两个最关键问题：

- CCG 的优势是不是稳定的？
- VMD / physics 的退化到底是偶然波动，还是系统性问题？

---

## 8. 这次已落地的产物

## 8.1 代码修复

- 已修改：`src/ship_motion/summarize_results.py`

## 8.2 新生成结果

- `results/result_1/outputs/summary/ablation_routine_test.csv`
- `results/result_1/outputs/summary/ablation_ood_test.csv`
- `results/result_1/outputs/summary/physics_metrics.csv`

## 8.3 建议你下一步优先做什么

如果你要我继续往下做，我建议下一步按这个顺序：

1. **补跑 persistence**
2. **我帮你生成正式图表（柱状图 / 预测曲线图）**
3. **我再把 VMD / physics 配置调成一版更保守的候选配置，并补充新的 smoke/正式命令**


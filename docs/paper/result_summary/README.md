# Paper Result Summary

## 1. 使用范围与目录说明

这份目录用于集中整理论文书写阶段要直接引用的实验结果材料，覆盖：

- `result_1`：基础模型对比与 CCG-xLSTM baseline 确立。
- `result_2_followup`：VMD 流程补跑与早期 follow-up 结果。
- `result3`：Phys / VMD 独立消融阶段。
- `result4`：单 seed 主实验筛选。
- `result4_next_round`：下一轮精简实验与候选收敛。
- `result4_robust`：6 个 shortlist 配置的 5-seed 稳健性验证。

生成文件：

- `data/representative_runs.csv`：跨阶段代表性模型总表。
- `data/best_by_stage.csv`：每一阶段的最佳 OOD 模型摘要。
- `data/result4_robust_seed_metrics.csv`：robust 30 次实验逐 seed 指标。
- `data/result4_robust_group_stats.csv`：robust 分组均值/方差统计。
- `data/result4_final_shortlist_robust.csv`：建议写入论文主文的最终 shortlist。
- `figures/stage_best_ood_rmse.png`：各阶段最佳 OOD RMSE 走势。
- `figures/result4_next_round_ood_rmse.png`：next_round 单 seed OOD 对比。
- `figures/result4_robust_ood_mean_std.png`：robust 的均值±标准差对比。
- `figures/result4_robust_tradeoff_jump.png`：预测性能与轨迹首步跳变的折中图。

## 2. 跨阶段主线结论

- `result_1` 中 `CCG-xLSTM Baseline` 的 OOD RMSE 为 `0.182219`，优于 LSTM / GRU / Lite-xLSTM / Transformer，也优于当时的早期 VMD 原型。这一阶段的核心价值是确定 baseline，而不是直接证明 VMD/Phys 有效。
- `result_2_followup` 的最佳 OOD 结果来自 `VMD-CCG-Phys-xLSTM Follow-up`，OOD RMSE `0.195851`。这组结果主要证明缓存化 VMD 流程已经打通，但仍属于历史中间态，不建议作为最终主结论。
- `result3` 首次把 Phys 和 VMD 拆开分析。Phys 最优候选 `e3_ccg_phys_s001_r005` 的 OOD RMSE 为 `0.180006`，VMD 最优候选 `e3_vmd_ccg_l005_lr5e4` 的 OOD RMSE 为 `0.181180`，说明两条方向都值得继续，但作用机制不同。
- `result4 main` 中，VMD 主线进一步收敛到 `e4_vmd_ccg_l005_lr3e4_nomix`，OOD RMSE `0.180483`，相对同阶段 baseline 改善明显。
- `result4 next_round` 中，联合方案 `e4_joint_l005_lr3e4_nomix_r0050` 取得当前单 seed 最低 OOD RMSE `0.180341`，说明 Joint 已经超过单纯 VMD 候选。
- `result4 robust` 的 5-seed 统计进一步确认了这一点：当前最优 robust OOD 均值来自 `Joint (l=0.05, lr=3e-4, r=0.005)`，为 `0.180640 ± 0.000623`。

## 3. Result4 最终可写结论

- baseline 的 robust OOD 均值为 `0.191022 ± 0.007513`。
- Phys 代表配置 `e4_ccg_phys_s0010_r0075_prphi` 的 robust OOD 均值为 `0.188913 ± 0.006159`，预测误差仍不占优，但其 `first_step_jump_rmse` 从 baseline 的 `0.184482` 显著降到 `0.007351`，`max_abs_second_diff_p95` 也从 `0.062500` 降到 `0.006622`，支持“限制不合理轨迹”的新叙事。
- VMD 主线的 robust 表现已经稳定优于 baseline，其中 `e4_vmd_ccg_l003_lr5e4_nomix` 是更好的 VMD 代表方案。
- Joint 是当前最强主线。`e4_joint_l005_lr3e4_nomix_r0050` 的 OOD 均值最低，而 `e4_joint_l005_lr3e4_nomix_s0010_r0050` 在 routine/val 和物理叙事上更完整。

## 4. 建议用于论文主文的最终 shortlist

| Model | OOD RMSE mean | OOD RMSE std | Routine RMSE mean | Val RMSE mean | Gain vs baseline (%) |
| --- | --- | --- | --- | --- | --- |
| Joint (l=0.05, lr=3e-4, r=0.005) | 0.180640 | 0.000623 | 0.024267 | 0.024096 | 5.435114 |
| Joint (l=0.05, lr=3e-4, s=0.001, r=0.005) | 0.180745 | 0.000575 | 0.024140 | 0.023970 | 5.379914 |
| VMD-CCG (l=0.03, lr=5e-4, no mixer) | 0.181291 | 0.001076 | 0.024537 | 0.024334 | 5.094175 |
| CCG-Phys (p/phi smooth, r=0.0075) | 0.188913 | 0.006159 | 0.030550 | 0.030111 | 1.104085 |
| CCG Baseline | 0.191022 | 0.007513 | 0.030712 | 0.030353 | 0.000000 |

推荐写法：

- 若主文只放 4 个代表模型：`baseline + phys_prphi + vmd_l003 + joint_s0010_r0050`。
- 若主文允许 5 个模型：再加入 `joint_r0050`，专门说明“最低 OOD 均值”和“更完整物理叙事”的两种联合 winner。

## 5. 文章写作时的引用建议

- 写基础模型对比时：优先引用 `data/representative_runs.csv` 里 `result_1` 的各模型行。
- 写 Phys / VMD 作用分离时：优先引用 `result3` 的代表行，并结合现有 `docs/result3_analysis_report.md`。
- 写实验四主结果时：优先引用 `data/result4_final_shortlist_robust.csv` 和 `figures/result4_robust_ood_mean_std.png`。
- 写 Phys 的新定位时：优先引用 `figures/result4_robust_tradeoff_jump.png`，强调 Phys 主要改善轨迹跳变与物理一致性，而不是直接夺取最低预测 RMSE。

## 6. 注意事项

- `result_2_followup` 的目录结构与其他阶段不同，正式结果位于 `results/result_2_followup/results/result_2_followup/outputs/`。本汇总已按该路径读取。
- 本目录中的图表和 CSV 由 `python -m ship_motion.paper_result_summary` 自动生成；若后续补跑实验，请重新执行该命令刷新材料。

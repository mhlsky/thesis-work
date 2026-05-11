# Step 07：消融实验、结果汇总与绘图

## 这一步的目标（给小白看的说明）

这一步不再新增模型，而是把前面做出来的模型系统地跑一遍，整理成论文能用的实验结果。

写论文不能只说“我的模型更好”，必须有表格和图证明。消融实验的作用是把每个改进模块一个一个加上去，让读者看到：CCG 带来多少提升，VMD 带来多少提升，注意力带来多少提升，物理约束和连续解码带来什么改善。

这一步完成后，你应该得到：

1. 一张 routine test 对比表；
2. 一张 OOD test 对比表；
3. 一张物理指标表；
4. `u` 和 `phi` 的预测曲线图；
5. RMSE 柱状图；
6. 可直接放进论文实验章节的结果文件。

这一步不要改模型结构，除非发现明显 bug。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/summarize_results.py
ship_motion/plot_results.py
configs/ablation_list.yaml
scripts/run_ablation.ps1
scripts/evaluate_all.ps1
scripts/plot_results.ps1
```

输出：

```text
outputs/summary/
  ablation_routine_test.csv
  ablation_ood_test.csv
  physics_metrics.csv
  figures/
    pred_u_routine.png
    pred_phi_routine.png
    pred_u_ood.png
    pred_phi_ood.png
    rmse_bar.png
    roll_consistency_bar.png
```

---

## 2. 必跑实验列表

| 顺序 | run_name | config | 说明 |
|---|---|---|---|
| 0 | persistence | 无需训练 | 最低基线 |
| 1 | lstm_seq128_pred10 | `configs/lstm.yaml` | 常规 LSTM |
| 2 | gru_seq128_pred10 | `configs/gru.yaml` | 轻量循环网络基线 |
| 3 | tcn_seq128_pred10 | `configs/tcn.yaml` | 卷积时序基线 |
| 4 | transformer_seq128_pred10 | `configs/transformer.yaml` | 标准注意力基线，可选 |
| 5 | lite_xlstm_seq128_pred10 | `configs/lite_xlstm.yaml` | xLSTM 指数门控主干 |
| 6 | ccg_xlstm_seq128_pred10 | `configs/ccg_xlstm.yaml` | 控制/风场条件门控 |
| 7 | vmd_ccg_xlstm_seq128_pred10 | `configs/vmd_ccg_xlstm.yaml` | VMD 辅助监督 |
| 8 | vmd_ccg_attn_xlstm_seq128_pred10 | `configs/vmd_ccg_attn_xlstm.yaml` | 线性注意力 |
| 9 | vmd_ccg_attn_phys_xlstm_seq128_pred10 | `configs/vmd_ccg_attn_phys_xlstm.yaml` | 最终模型 |

如果时间不够，最低跑：

```text
persistence
lstm
gru
tcn
lite_xlstm
vmd_ccg_xlstm
vmd_ccg_attn_phys_xlstm
```

---

## 3. 指标表

每个 run 都应有：

```text
metrics_routine_test.json
metrics_ood_test.json
```

汇总成 CSV：

```text
model, mae_mean, rmse_mean, r2_mean, rmse_u, rmse_v, rmse_p, rmse_r, rmse_phi
```

论文主表建议只放：

- MAE_mean
- RMSE_mean
- R²_mean
- RMSE_u
- RMSE_phi

完整各变量指标可放附表。

---

## 4. 物理指标表

输出：

```text
model, split, smoothness, roll_consistency_rmse
```

论文中重点比较：

```text
VMD-CCG-Attn-xLSTM vs VMD-CCG-Attn-Phys-xLSTM
```

如果最终模型误差只小幅提升，但物理一致性明显更好，也可以成立。物理约束本来就不只是为了降低 MSE，而是为了减少反物理预测。

---

## 5. 可选附加消融

如果主实验结果已经跑通，可补充两个小消融：

| 实验 | 目的 |
|---|---|
| Final w/o Delta Decoder | 验证增量解码是否减少跳变 |
| Final w/o State Mixer | 验证状态耦合修正是否有效 |

这两个不是最低必做，但很适合写论文讨论。

---

## 6. 预测曲线图

评估时建议保存部分预测：

```text
outputs/{run_name}/predictions_routine_test.npz
outputs/{run_name}/predictions_ood_test.npz
```

内容：

```python
y_true: [N, pred_len, 5]
y_pred: [N, pred_len, 5]
last_state_raw: [N, 5]
```

画图时建议取：

```text
horizon = 0   # 预测未来第 1 秒
变量：u、phi
窗口数：前 200 个窗口
```

一张图中画：

- True
- LSTM
- GRU
- TCN
- Lite-xLSTM
- VMD-CCG-xLSTM
- Final Model

---

## 7. 多步长补充实验（可选但推荐）

主实验完成后，增加：

```text
pred_len = 1, 5, 10, 30
```

只跑两个模型：

1. LSTM；
2. VMD-CCG-Attn-Phys-xLSTM。

这样能证明最终模型在预测步长变长时更稳定。

---

## 8. 脚本设计

### 8.1 `run_ablation.ps1`

顺序执行：

```powershell
python -m ship_motion.train --config configs/lstm.yaml
python -m ship_motion.train --config configs/gru.yaml
python -m ship_motion.train --config configs/tcn.yaml
python -m ship_motion.train --config configs/transformer.yaml
python -m ship_motion.train --config configs/lite_xlstm.yaml
python -m ship_motion.train --config configs/ccg_xlstm.yaml
python -m ship_motion.train --config configs/vmd_ccg_xlstm.yaml
python -m ship_motion.train --config configs/vmd_ccg_attn_xlstm.yaml
python -m ship_motion.train --config configs/vmd_ccg_attn_phys_xlstm.yaml
python -m ship_motion.summarize_results --runs lstm_seq128_pred10,gru_seq128_pred10,tcn_seq128_pred10,transformer_seq128_pred10,lite_xlstm_seq128_pred10,ccg_xlstm_seq128_pred10,vmd_ccg_xlstm_seq128_pred10,vmd_ccg_attn_xlstm_seq128_pred10,vmd_ccg_attn_phys_xlstm_seq128_pred10
```

如果训练慢，可以一个一个手动跑，不强求一键跑完。

---

## 9. 验收标准

必须满足：

1. 主实验都有 routine test 和 OOD test 指标；
2. 汇总 CSV 能生成；
3. 至少生成 `u` 和 `phi` 预测曲线图；
4. 最终模型与基线有可解释提升；
5. 图和表可以直接用于论文。

---

## 10. 交给 AI 编码的提示词

```text
请根据 implementation_steps/07_experiments.md 实现实验汇总和绘图脚本。不要新增模型。要求读取各 run 的 metrics JSON 和 predictions npz，生成 routine/OOD 消融表、物理指标表、预测曲线图和 RMSE 柱状图。
```



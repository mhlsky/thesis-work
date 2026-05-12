# Step 07：主线消融实验、结果汇总与绘图

## 这一步的目标（给小白看的说明）

这一步不再新增模型，而是把前面做出来的主线模型系统地跑一遍，整理成论文能用的实验结果。

写论文不能只说“我的模型更好”，必须有表格和图证明。消融实验的作用是把每个改进模块一个一个加上去，让读者看到：CCG 带来多少提升，VMD 带来多少提升，物理约束和连续解码带来什么改善。

注意：线性注意力已经降级为可选增强，不属于本步骤主线必跑项。如果主线实验完成后还有时间，再按 `08_optional_attention.md` 补充。

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

## 2. 主线必跑实验列表

| 顺序 | run_name | config | 说明 |
|---|---|---|---|
| 0 | persistence | 无需训练 | 最低基线 |
| 1 | lstm_seq128_pred10 | `configs/lstm.yaml` | 常规 LSTM |
| 2 | gru_seq128_pred10 | `configs/gru.yaml` | 轻量循环网络基线 |
| 3 | transformer_seq128_pred10 | `configs/transformer.yaml` | 标准注意力基线 |
| 4 | lite_xlstm_seq128_pred10 | `configs/lite_xlstm.yaml` | xLSTM 指数门控主干 |
| 5 | ccg_xlstm_seq128_pred10 | `configs/ccg_xlstm.yaml` | 控制/风场条件门控 |
| 6 | vmd_ccg_xlstm_seq128_pred10 | `configs/vmd_ccg_xlstm.yaml` | VMD 辅助监督 |
| 7 | vmd_ccg_phys_xlstm_seq128_pred10 | `configs/vmd_ccg_phys_xlstm.yaml` | 最终主模型 |

可选：

| run_name | config | 说明 |
|---|---|---|
| tcn_seq128_pred10 | `configs/tcn.yaml` | 卷积时序基线，可选 |
| vmd_ccg_attn_phys_xlstm_seq128_pred10 | `configs/vmd_ccg_attn_phys_xlstm.yaml` | 线性注意力增强，可选 |

如果时间不够，最低跑：

```text
persistence
lstm
gru
transformer
lite_xlstm
vmd_ccg_xlstm
vmd_ccg_phys_xlstm
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
VMD-CCG-xLSTM vs VMD-CCG-Phys-xLSTM
```

如果最终模型误差只小幅提升，但物理一致性明显更好，也可以成立。物理约束本来就不只是为了降低 MSE，而是为了减少反物理预测。

---

## 5. 可选附加消融

如果主实验结果已经跑通，可补充三个小消融：

| 实验 | 目的 |
|---|---|
| Final w/o Delta Decoder | 验证增量解码是否减少跳变 |
| Final w/o State Mixer | 验证状态耦合修正是否有效 |
| Final + Linear Attention | 验证线性注意力是否带来额外收益 |

这些不是最低必做。

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
- Transformer
- Lite-xLSTM
- VMD-CCG-xLSTM
- VMD-CCG-Phys-xLSTM

---

## 7. 多步长补充实验（可选但推荐）

主实验完成后，增加：

```text
pred_len = 1, 5, 10, 30
```

只跑两个模型：

1. LSTM；
2. VMD-CCG-Phys-xLSTM。

这样能证明最终模型在预测步长变长时更稳定。

---

## 8. 脚本设计

### 8.1 `run_ablation.ps1`

顺序执行：

```powershell
python -m ship_motion.train --config configs/lstm.yaml
python -m ship_motion.train --config configs/gru.yaml
python -m ship_motion.train --config configs/transformer.yaml
python -m ship_motion.train --config configs/lite_xlstm.yaml
python -m ship_motion.train --config configs/ccg_xlstm.yaml
python -m ship_motion.train --config configs/vmd_ccg_xlstm.yaml
python -m ship_motion.train --config configs/vmd_ccg_phys_xlstm.yaml
python -m ship_motion.summarize_results --runs lstm_seq128_pred10,gru_seq128_pred10,transformer_seq128_pred10,lite_xlstm_seq128_pred10,ccg_xlstm_seq128_pred10,vmd_ccg_xlstm_seq128_pred10,vmd_ccg_phys_xlstm_seq128_pred10
```

如果训练慢，可以一个一个手动跑，不强求一键跑完。

---

## 9. 验收标准

必须满足：

1. 主实验都有 routine test 和 OOD test 指标；
2. 汇总 CSV 能生成；
3. 至少生成 `u` 和 `phi` 预测曲线图；
4. 最终主模型与基线有可解释提升；
5. 图和表可以直接用于论文。

---

## 10. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/07_experiments.md 实现实验汇总和绘图脚本。不要新增模型。要求读取各 run 的 metrics JSON 和 predictions npz，生成 routine/OOD 消融表、物理指标表、预测曲线图和 RMSE 柱状图。线性注意力只作为可选增强，不纳入主线必跑项。
```


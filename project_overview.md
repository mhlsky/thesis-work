# 基于 VMD-CCG-xLSTM 的船舶运动姿态预测实验项目总览

> 本文档用于指导后续 AI 从零编写实验代码。目标是完成一套结构清楚、能稳定训练、能支撑 EI 论文实验的船舶姿态预测代码。
>
> 当前推荐模型主线：**VMD + 控制/风场条件门控 xLSTM + 线性注意力 + 增量解码 + 状态耦合 Mixer + 物理约束**。
>
> 简称可写为：`VMD-CCG-Attn-Phys-xLSTM`。其中 `CCG` 表示 Control-Conditioned Gating，即控制/环境条件门控。

---

## 1. 项目一句话说明

用船舶仿真时序数据，训练一个面向海事仿真场景的多步预测模型。模型利用历史控制量、风场量和船舶运动状态，预测未来若干秒的：

```text
u, v, p, r, phi
```

并通过消融实验证明以下模块有效：

1. xLSTM 风格指数门控比普通 LSTM 更适合长时序船舶运动预测；
2. 控制/风场条件门控能让模型根据转速、舵角、风速变化自适应调整记忆；
3. VMD 分解辅助监督能降低趋势、周期、扰动混合带来的学习难度；
4. 线性注意力能帮助模型关注关键历史时刻；
5. 增量解码和物理约束能减少姿态跳变，使预测结果更符合船舶运动连续性；
6. 状态耦合 Mixer 能建模 `u,v,p,r,phi` 之间的多自由度耦合关系。

---

## 2. 数据与任务定义

### 2.1 数据目录

```text
data/patrol_ship_routine/processed/train/*.csv        # 训练集
data/patrol_ship_routine/processed/validation/*.csv   # 验证集
data/patrol_ship_routine/processed/test/*.csv         # 常规测试集
data/patrol_ship_ood/processed/test/*.csv             # OOD 分布外测试集
```

每个 CSV 约 1 小时，采样频率 1 Hz。滑动窗口只能在单个 CSV 内部生成，不能跨文件拼接。

### 2.2 字段分组

完整字段：

```text
time,n,deltal,deltar,Vw,alpha_x,alpha_y,u,v,p,r,phi
```

建议明确分成两类输入：

```python
exog_cols = ["n", "deltal", "deltar", "Vw", "alpha_x", "alpha_y"]
state_cols = ["u", "v", "p", "r", "phi"]
input_cols = exog_cols + state_cols
target_cols = state_cols
```

含义：

- `exog_cols` 是外生驱动量：推进器、舵角、风场；
- `state_cols` 是船舶自身运动状态；
- 预测目标是未来的 `state_cols`。

### 2.3 时序预测形式

默认主实验：

```text
seq_len = 128       # 看过去 128 秒
pred_len = 10       # 预测未来 10 秒
输入 x: [batch, 128, 11]
输出 y: [batch, 10, 5]
```

补充实验可做：

```text
pred_len = 1, 5, 10, 30
```

---

## 3. 推荐最终模型结构

最终模型结构建议如下：

```text
历史输入 X
  ├─ exog:  n, deltal, deltar, Vw, alpha_x, alpha_y
  └─ state: u, v, p, r, phi

X -> 输入投影
  -> CCG-Lite-xLSTM 主干
       # 外生控制/风场量调节 xLSTM 门控
  -> Linear Self-Attention
       # 关注关键历史时刻
  -> VMD Multi-Head
       # 分别预测 K 个 VMD 模态
  -> 模态求和
  -> Delta Decoder
       # 预测未来增量并累积成状态
  -> State Coupling Mixer
       # 修正多状态耦合关系
  -> y_hat

Loss = prediction_loss
     + lambda_vmd * vmd_aux_loss
     + lambda_smooth * smoothness_loss
     + lambda_roll * roll_kinematic_loss
```

---

## 4. 必做模型与消融路线

### 4.1 主线模型

| 模型名 | 说明 | 论文作用 |
|---|---|---|
| Persistence | 未来状态复制历史最后一帧 | 最低基线 |
| LSTM | 标准循环网络 | 常规循环网络基线 |
| GRU | 比 LSTM 更轻量的门控循环网络 | 轻量 RNN 基线 |
| TCN | 时间卷积网络，使用因果/膨胀卷积建模时序 | 卷积时序基线 |
| Transformer Encoder | 标准自注意力时序模型 | 注意力基线，验证复杂度与精度权衡 |
| Lite-xLSTM | 指数门控 xLSTM 风格主干 | 验证 xLSTM 主干 |
| CCG-xLSTM | 控制/风场条件门控 xLSTM | 验证船舶场景特化门控 |
| VMD-CCG-xLSTM | VMD 辅助监督 + CCG-xLSTM | 验证 VMD |
| VMD-CCG-Attn-xLSTM | 加线性注意力 | 验证注意力 |
| VMD-CCG-Attn-Phys-xLSTM | 加增量解码、状态耦合、物理约束 | 最终模型 |

如果时间有限，最低保留：

```text
Persistence, LSTM, GRU, TCN, Lite-xLSTM, VMD-CCG-xLSTM, VMD-CCG-Attn-Phys-xLSTM
```

Transformer Encoder 建议作为补充基线：如果训练时间充足则加入主表；如果时间不足，可放入附表或说明由于计算复杂度较高只做代表性对比。

### 4.2 推荐消融表

| 实验 | CCG | VMD | Attention | Delta Decoder | State Mixer | Physics | 目的 |
|---|---|---|---|---|---|---|---|
| LSTM | 否 | 否 | 否 | 否 | 否 | 否 | 常规循环网络基线 |
| GRU | 否 | 否 | 否 | 否 | 否 | 否 | 轻量循环网络基线 |
| TCN | 否 | 否 | 否 | 否 | 否 | 否 | 卷积时序基线 |
| Transformer | 否 | 否 | 标准注意力 | 否 | 否 | 否 | 注意力基线，可选 |
| Lite-xLSTM | 否 | 否 | 否 | 可关 | 可关 | 否 | xLSTM 主干 |
| CCG-xLSTM | 是 | 否 | 否 | 可关 | 可关 | 否 | 控制条件门控 |
| VMD-CCG-xLSTM | 是 | 是 | 否 | 可关 | 可关 | 否 | VMD 有效性 |
| VMD-CCG-Attn-xLSTM | 是 | 是 | 是 | 可关 | 可关 | 否 | 注意力有效性 |
| Final | 是 | 是 | 是 | 是 | 是 | 是 | 最终模型 |

说明：

- `Delta Decoder` 和 `State Mixer` 实现成本低，建议放入最终模型；
- 如果结果需要更细，可额外做 `Final w/o Delta`、`Final w/o Mixer`；
- EI 论文主表不必塞太多模型，附表可以补充。

---

## 5. 创新点 1：控制/风场条件门控 CCG

### 5.1 为什么适合本课题

船舶运动不是单纯由过去状态决定，还强烈受以下因素驱动：

```text
n, deltal, deltar, Vw, alpha_x, alpha_y
```

这些外生量决定船舶接下来是加速、转向，还是受到风浪扰动。因此 xLSTM 的记忆写入和遗忘不应只依赖历史状态，还应由控制和环境量调节。

### 5.2 实现思路

对每个时间步，将输入拆成：

```python
exog_t = [n, deltal, deltar, Vw, alpha_x, alpha_y]
state_t = [u, v, p, r, phi]
```

先用小 MLP 编码外生量：

```text
context_t = MLP(exog_t)
```

然后让 `context_t` 参与 xLSTM 门控：

```text
i_pre = W_i[state_t, h_{t-1}] + U_i context_t
f_pre = W_f[state_t, h_{t-1}] + U_f context_t
o_pre = W_o[state_t, h_{t-1}] + U_o context_t
z_pre = W_z[state_t, h_{t-1}] + U_z context_t
```

再使用指数门控：

```text
i_t = exp(clamp(i_pre, -gate_clip, gate_clip))
f_t = exp(clamp(f_pre, -gate_clip, gate_clip))
```

论文表述：

> 针对船舶运动受推进器、舵角和风场强驱动的特点，本文设计控制/环境条件门控机制，将外生航行控制量引入 xLSTM 指数门控过程，使模型能够根据工况变化自适应调整记忆更新强度。

---

## 6. 创新点 2：VMD 分解辅助监督

VMD 是必做模块。推荐只作为辅助监督标签，不作为模型输入，避免未来信息泄漏。

模型输出 K 个模态：

```text
mode_preds: [B, pred_len, target_dim, K]
y_hat_base = sum(mode_preds over K)
```

训练损失：

```text
loss_vmd = MSE(mode_preds, y_modes)
loss_pred = MSE(y_hat, y_true)
```

总损失中加入：

```text
lambda_vmd * loss_vmd
```

推荐参数：

```yaml
vmd:
  K: 3
  alpha: 2000
  lambda_vmd: 0.2
```

---

## 7. 创新点 3：增量解码 Delta Decoder

直接预测未来绝对状态容易产生跳变。船舶运动具有连续性，因此建议模型预测未来每一步的增量：

```text
Δy_1, Δy_2, ..., Δy_H
```

然后从历史最后状态开始累积：

```text
y_hat_1 = y_last + Δy_1
y_hat_2 = y_hat_1 + Δy_2
...
y_hat_H = y_hat_{H-1} + Δy_H
```

优点：

- 自然符合连续运动；
- 对 `phi` 这种姿态角更稳定；
- 与平滑约束、横摇运动学约束配合更好。

实现建议：

```yaml
model:
  decode_type: delta     # direct | delta
```

---

## 8. 创新点 4：状态耦合 Mixer

预测目标 `u,v,p,r,phi` 之间不是独立的。可以在每个预测步上加入轻量状态耦合层：

```text
y_hat_t = y_hat_t + MLP_or_Linear(y_hat_t)
```

最简单实现：

```python
nn.Linear(target_dim, target_dim)
```

作用：让模型学习多自由度输出之间的相互修正，例如：

- `p` 与 `phi` 的关系；
- `v` 与 `r` 的横荡/艏摇耦合；
- 控制输入变化对多个状态的联合影响。

配置：

```yaml
model:
  use_state_mixer: true
```

---

## 9. 创新点 5：线性注意力

在 CCG-xLSTM 输出序列后加入线性注意力：

```text
hidden_seq -> LinearSelfAttention -> feature
```

作用：关注舵角突变、风速突变、横摇峰值等关键历史时刻。

不要使用标准 Transformer 注意力作为主模块，避免复杂度过高。线性注意力用 `elu(x)+1` 核映射即可。

---

## 10. 创新点 6：物理约束

先做两类稳妥约束。

### 10.1 平滑约束

```text
loss_smooth = mean((y_{t+1} - 2y_t + y_{t-1})^2)
```

### 10.2 横摇运动学一致性

因为有 `p` 和 `phi`：

```text
phi_{t+1} - phi_t ≈ p_t * dt
```

总损失：

```text
loss = pred_loss
     + lambda_vmd * vmd_loss
     + lambda_smooth * smooth_loss
     + lambda_roll * roll_kinematic_loss
```

推荐：

```yaml
lambda_smooth: 0.01
lambda_roll: 0.05
```

---

## 11. 推荐代码结构

```text
thesis-work/
  project_overview.md
  learn.md
  implementation_steps/
  configs/
    base.yaml
    lstm.yaml
    gru.yaml
    tcn.yaml
    transformer.yaml
    lite_xlstm.yaml
    ccg_xlstm.yaml
    vmd_ccg_xlstm.yaml
    vmd_ccg_attn_xlstm.yaml
    vmd_ccg_attn_phys_xlstm.yaml
  ship_motion/
    data/
      dataset.py
      scaler.py
      vmd.py
    models/
      persistence.py
      lstm.py
      gru.py
      tcn.py
      transformer.py
      xlstm.py              # LiteXLSTM + CCGXLSTM
      linear_attention.py
      vmd_heads.py
      decoders.py           # DeltaDecoder
      state_mixer.py
    losses/
      vmd_loss.py
      physics.py
    train.py
    evaluate.py
    metrics.py
    plot_results.py
    summarize_results.py
    utils.py
  scripts/
    smoke_test.ps1
    build_vmd_cache.ps1
    run_baselines.ps1
    run_ablation.ps1
```

---
## 12. 分步骤编码路线

1. `01_data_pipeline.md`：数据读取、滑窗、标准化，明确 `exog_cols` 和 `state_cols`；
2. `02_training_lstm.md`：Persistence、LSTM、GRU、TCN、Transformer Encoder、训练评估框架；
3. `03_vmd_module.md`：VMD 分解、缓存、Dataset 返回 `y_modes`；
4. `04_xlstm_backbone.md`：Lite-xLSTM 与 CCG-xLSTM 主干；
5. `05_vmd_ccg_integration.md`：VMD 多分支头、Delta Decoder、State Mixer；
6. `06_attention_physics.md`：线性注意力、物理约束、最终模型配置；
7. `07_experiments.md`：消融实验、汇总表格、绘图。

---

## 13. EI 论文最低完成标准

最低需要产出：

1. 数据集和任务定义；
2. Persistence、LSTM、GRU、TCN、Transformer Encoder（可选）、Lite-xLSTM、CCG-xLSTM、VMD-CCG-xLSTM、Final 结果；
3. routine test 和 OOD test 两套指标；
4. MAE、RMSE、R² 表格；
5. VMD / CCG / Attention / Physics 消融表；
6. `u` 和 `phi` 预测曲线图；
7. 物理一致性指标：roll consistency error 或 smoothness；
8. 对 OOD 泛化结果进行分析。




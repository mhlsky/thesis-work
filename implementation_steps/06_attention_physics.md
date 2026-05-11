# Step 06：线性注意力与物理约束，形成最终模型

## 这一步的目标（给小白看的说明）

这一步在 VMD-CCG-xLSTM 的基础上加入最后两个模块：**线性注意力** 和 **物理约束损失**，形成最终模型：

```text
VMD-CCG-Attn-Phys-xLSTM
```

前面已经解决了：

- CCG：让控制量和风场量参与门控；
- VMD：让模型按趋势、周期、扰动分量学习；
- Delta Decoder：让预测更连续；
- State Mixer：让多状态输出之间互相修正。

这一步进一步解决：

1. 历史 128 秒里哪些时刻更关键；
2. 预测结果是否符合船舶运动的连续性和横摇运动学关系。

这一步完成后，你应该得到两个实验组：

1. `VMD-CCG-Attn-xLSTM`：验证线性注意力；
2. `VMD-CCG-Attn-Phys-xLSTM`：最终模型。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/models/linear_attention.py
ship_motion/losses/physics.py
configs/vmd_ccg_attn_xlstm.yaml
configs/vmd_ccg_attn_phys_xlstm.yaml
scripts/train_final_model.ps1
```

修改：

```text
ship_motion/models/xlstm.py       # encode 后可选 attention
ship_motion/train.py              # 支持 physics loss
ship_motion/metrics.py            # 增加物理指标
```

---

## 2. 线性注意力设计

### 2.1 放在哪里

放在 CCG-xLSTM 输出序列之后、VMD head 之前：

```text
Input -> CCG-xLSTM hidden_seq -> LinearSelfAttention -> feature -> VMDMultiHead
```

这样注意力能在历史窗口中挑选关键时间片，例如：

- 舵角突变；
- 转速快速变化；
- 风速突变；
- 横摇角速度峰值。

### 2.2 接口

`ship_motion/models/linear_attention.py`：

```python
class LinearSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, eps=1e-6):
        pass

    def forward(self, x):
        # x: [B, L, D]
        # return: [B, L, D]
        pass
```

### 2.3 算法

使用 `elu(x) + 1` 核映射：

```python
q = elu(q) + 1
k = elu(k) + 1
kv = einsum(k, v)
z = 1 / (einsum(q, sum(k)) + eps)
out = einsum(q, kv) * z
```

外面加：

```text
Linear output projection + Dropout + Residual + LayerNorm
```

---

## 3. 接入 VMD-CCG-xLSTM

模型配置：

```yaml
model:
  use_attention: true
  attn_heads: 4
```

forward 逻辑：

```python
hidden_seq, _ = backbone.encode(x, batch=batch)
if use_attention:
    hidden_seq = attention(hidden_seq)
feature = hidden_seq[:, -1]
mode_preds = vmd_head(feature)
...
```

---

## 4. 物理约束损失

物理损失必须在反标准化后的真实物理尺度上计算。

### 4.1 平滑性约束

船舶运动有惯性，未来预测序列不应有突然尖峰。使用二阶差分：

```python
d1 = y[:, 1:, :] - y[:, :-1, :]
d2 = d1[:, 1:, :] - d1[:, :-1, :]
loss_smooth = mean(d2 ** 2)
```

### 4.2 横摇运动学约束

目标顺序：

```python
target_cols = ["u", "v", "p", "r", "phi"]
p_idx = 2
phi_idx = 4
```

1 Hz 下：

```text
phi_{t+1} - phi_t ≈ p_t * dt
```

预测序列内部：

```python
phi_hat[:, 1:] - phi_hat[:, :-1] ≈ p_hat[:, :-1] * dt
```

连接历史最后一帧：

```python
phi_hat[:, 0] - phi_last ≈ p_last * dt
```

### 4.3 接口

`ship_motion/losses/physics.py`：

```python
def smoothness_loss(y_raw): pass

def roll_kinematic_loss(y_raw, last_state_raw, dt=1.0, p_idx=2, phi_idx=4): pass

def physics_loss(y_raw, last_state_raw, lambda_smooth, lambda_roll, dt=1.0): pass
```

---

## 5. 总损失

最终模型训练损失：

```text
loss = pred_loss
     + lambda_vmd * vmd_loss
     + lambda_smooth * smoothness_loss
     + lambda_roll * roll_kinematic_loss
```

推荐初始值：

```yaml
vmd:
  lambda_vmd: 0.2
physics:
  enabled: true
  lambda_smooth: 0.01
  lambda_roll: 0.05
  dt: 1.0
```

如果训练变差，先降低：

```text
lambda_smooth = 0.001
lambda_roll = 0.01
```

---

## 6. 配置文件

### 6.1 `configs/vmd_ccg_attn_xlstm.yaml`

```yaml
run_name: vmd_ccg_attn_xlstm_seq128_pred10
model:
  name: vmd_ccg_xlstm
  use_attention: true
  attn_heads: 4
  decode_type: delta
  use_state_mixer: true
vmd:
  enabled: true
  K: 3
  lambda_vmd: 0.2
physics:
  enabled: false
```

完整配置可复制 `vmd_ccg_xlstm.yaml`。

### 6.2 `configs/vmd_ccg_attn_phys_xlstm.yaml`

```yaml
run_name: vmd_ccg_attn_phys_xlstm_seq128_pred10
model:
  name: vmd_ccg_xlstm
  use_attention: true
  attn_heads: 4
  decode_type: delta
  use_state_mixer: true
vmd:
  enabled: true
  K: 3
  lambda_vmd: 0.2
physics:
  enabled: true
  lambda_smooth: 0.01
  lambda_roll: 0.05
  dt: 1.0
```

---

## 7. 物理指标

除了训练损失，还要在测试时输出物理指标：

```python
def smoothness_metric(y_pred_raw): pass

def roll_consistency_rmse(y_pred_raw, last_state_raw, dt=1.0): pass
```

论文里可以用这些指标证明：最终模型不仅误差低，而且更符合运动规律。

---

## 8. 调试顺序

不要一次全开。按顺序：

1. VMD-CCG-xLSTM 正常训练；
2. 打开线性注意力，得到 VMD-CCG-Attn-xLSTM；
3. 只开 `lambda_smooth=0.001`；
4. 再开 `lambda_roll=0.01`；
5. 最后使用正式权重。

---

## 9. 验收标准

必须满足：

1. `VMD-CCG-Attn-xLSTM` 能训练和评估；
2. `VMD-CCG-Attn-Phys-xLSTM` 能训练和评估；
3. 注意力输出 shape 不变；
4. 物理损失在 raw 尺度计算；
5. 最终模型至少在 OOD 或物理指标上优于无物理约束版本。

---

## 10. 交给 AI 编码的提示词

```text
请根据 implementation_steps/06_attention_physics.md，在已有 VMD-CCG-xLSTM 基础上加入 LinearSelfAttention 和物理约束损失。注意力和物理约束都要能通过配置开关控制。物理损失必须在反标准化后的 raw 物理尺度上计算。
```


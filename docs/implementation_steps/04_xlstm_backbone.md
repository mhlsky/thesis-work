# Step 04：Lite-xLSTM 与 CCG-xLSTM 主干模型

## 这一步的目标（给小白看的说明）

这一步开始实现项目的核心时序模型：**Lite-xLSTM** 和它的船舶场景增强版 **CCG-xLSTM**。

普通 LSTM 虽然能处理时序，但在长时间依赖、复杂扰动和深层训练稳定性上有限。xLSTM 的关键思想是使用更强的门控和记忆机制。本项目不强求复现完整论文版 xLSTM，而是实现一个适合本课题、容易调试的 Lite-xLSTM：使用指数门控和归一化记忆更新。

进一步地，船舶运动强烈受转速、舵角和风场影响。因此本步骤还要实现 **控制/风场条件门控 CCG**：让 `n,deltal,deltar,Vw,alpha_x,alpha_y` 直接参与 xLSTM 门控，决定模型什么时候记忆、什么时候遗忘。

这一步完成后，你应该能得到两个模型：

1. `Lite-xLSTM`：只使用指数门控主干；
2. `CCG-xLSTM`：在 Lite-xLSTM 基础上加入控制/环境条件门控。

这一步暂时不接 VMD、不接注意力、不接物理约束。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/models/xlstm.py
configs/lite_xlstm.yaml
configs/ccg_xlstm.yaml
scripts/train_lite_xlstm.ps1
scripts/train_ccg_xlstm.ps1
```

修改：

```text
ship_motion/train.py      # build_model 支持 lite_xlstm 和 ccg_xlstm
ship_motion/evaluate.py   # 支持加载两类模型
```

---

## 2. Lite-xLSTM 总体结构

```text
x: [B, L, 11]
  -> input_proj: Linear(11, d_model)
  -> LiteXLSTMBlockStack
  -> last hidden: [B, d_model]
  -> ForecastHead
  -> y_hat: [B, pred_len, 5]
```

默认参数：

```yaml
d_model: 128
num_layers: 2
dropout: 0.1
gate_clip: 5.0
```

---

## 3. Lite-xLSTM 单元公式

对每个时间步：

```text
i_pre, f_pre, z_pre, o_pre = Linear([x_t, h_{t-1}])

i_t = exp(clamp(i_pre, -gate_clip, gate_clip))
f_t = exp(clamp(f_pre, -gate_clip, gate_clip))
z_t = tanh(z_pre)
o_t = sigmoid(o_pre)

c_t = (f_t * c_{t-1} + i_t * z_t) / (f_t + i_t + eps)
h_t = o_t * tanh(c_t)
```

说明：

- `i_t` 控制新信息写入；
- `f_t` 控制旧记忆保留；
- 分母 `f_t + i_t + eps` 稳定数值；
- `gate_clip` 防止 `exp()` 爆炸。

---

## 4. CCG：控制/风场条件门控

### 4.1 为什么需要 CCG

船舶状态不是自己凭空变化，而是受外生量驱动：

```text
exog_t = [n, deltal, deltar, Vw, alpha_x, alpha_y]
state_t = [u, v, p, r, phi]
```

如果某一时刻舵角变化大、风速突增，模型应该更重视这些新信息。因此门控应该由 `exog_t` 调节。

### 4.2 CCG 公式

先编码外生量：

```text
context_t = MLP(exog_t)
```

再参与门控：

```text
gate_base = W [state_t, h_{t-1}]
gate_context = U context_t

i_pre, f_pre, z_pre, o_pre = gate_base + gate_context
```

之后继续使用 Lite-xLSTM 的指数门控公式。

### 4.3 实现建议

`CCGLiteXLSTMCell` 输入应明确拆分：

```python
def forward(self, x_state_t, x_exog_t, state):
    pass
```

模型 forward 可以使用 Dataset 返回的：

```python
batch["x_state"]
batch["x_exog"]
```

为了兼容旧接口，如果只传 `x`，也可以按列切片：

```python
x_exog = x[:, :, :6]
x_state = x[:, :, 6:]
```

---

## 5. 模块接口

```python
class LiteXLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, gate_clip=5.0): pass
    def forward(self, x_t, state): pass

class CCGLiteXLSTMCell(nn.Module):
    def __init__(self, state_dim, exog_dim, hidden_dim, context_dim=64, gate_clip=5.0): pass
    def forward(self, x_state_t, x_exog_t, state): pass

class LiteXLSTMForecaster(nn.Module):
    def forward(self, x, batch=None): pass

class CCGXLSTMForecaster(nn.Module):
    def forward(self, x, batch=None): pass
```

所有 forecaster 输出：

```text
[B, pred_len, 5]
```

---

## 6. 稳定性设计

每层输出后建议加：

```text
Dropout + Residual + LayerNorm
```

训练中必须开启：

```yaml
grad_clip: 1.0
```

如果出现 NaN：

1. `lr` 从 `1e-3` 降到 `5e-4`；
2. `gate_clip` 从 `5.0` 降到 `3.0`；
3. `context_dim` 从 `64` 降到 `32`；
4. 暂时把 `num_layers` 改成 1。

---

## 7. 配置文件

### 7.1 `configs/lite_xlstm.yaml`

```yaml
run_name: lite_xlstm_seq128_pred10
model:
  name: lite_xlstm
  input_dim: 11
  target_dim: 5
  pred_len: 10
  d_model: 128
  num_layers: 2
  dropout: 0.1
  gate_clip: 5.0
```

### 7.2 `configs/ccg_xlstm.yaml`

```yaml
run_name: ccg_xlstm_seq128_pred10
model:
  name: ccg_xlstm
  exog_dim: 6
  state_dim: 5
  target_dim: 5
  pred_len: 10
  d_model: 128
  context_dim: 64
  num_layers: 2
  dropout: 0.1
  gate_clip: 5.0
```

完整配置可以复制 `lstm.yaml`，只替换 `model` 部分和 `run_name`。

---

## 8. 验收标准

必须满足：

1. Lite-xLSTM 输入 `[B,128,11]`，输出 `[B,10,5]`；
2. CCG-xLSTM 能使用 `x_exog` 和 `x_state`；
3. 两个模型都能训练 1 个 epoch 无 NaN；
4. routine test 和 OOD test 有指标；
5. CCG-xLSTM 至少不明显差于 Lite-xLSTM，最好在 OOD 上更稳。

---

## 9. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/04_xlstm_backbone.md 实现 Lite-xLSTM 和 CCG-xLSTM。Lite-xLSTM 使用指数门控和归一化记忆更新；CCG-xLSTM 需要让 exog_cols 经过 MLP 后参与门控。暂时不要接 VMD、注意力和物理约束。模型输出必须是 [B, pred_len, 5]。
```



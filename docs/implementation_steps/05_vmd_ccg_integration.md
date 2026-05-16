# Step 05：VMD-CCG-xLSTM 集成、增量解码与状态耦合 Mixer

## 这一步的目标（给小白看的说明）

这一步把前面两个关键部分合起来：

1. Step 03 做好的 VMD 分解标签；
2. Step 04 做好的 CCG-xLSTM 主干。

目标是形成第一个比较完整的创新模型：**VMD-CCG-xLSTM**。

直观理解：模型不再直接预测一个复杂的船舶运动序列，而是先通过 VMD 多分支头分别预测“趋势、周期、扰动”等模态，再把这些模态加起来。为了让预测更符合船舶连续运动，本步骤还加入两个低成本但很实用的结构：

- **Delta Decoder**：预测未来状态增量，而不是直接预测绝对值；
- **State Coupling Mixer**：让 `u,v,p,r,phi` 之间可以相互修正，体现多自由度耦合。

这一步完成后，你应该能训练：

```text
VMD-CCG-xLSTM
```

这一步暂时不做线性注意力和物理损失，它们放到下一步。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/models/vmd_heads.py
ship_motion/models/decoders.py
ship_motion/models/state_mixer.py
configs/vmd_ccg_xlstm.yaml
scripts/smoke_step_05_vmd_ccg.ps1
scripts/train_vmd_ccg_xlstm.ps1
```

修改文件：

```text
ship_motion/models/xlstm.py    # CCG backbone 支持 encode() 返回特征
ship_motion/train.py           # 支持 vmd loss 和 dict output
ship_motion/evaluate.py        # 支持 VMD 模型加载与评估
```

Smoke test 要求：

- `scripts/smoke_step_05_vmd_ccg.ps1` 必须验证 CCG backbone、VMD head、Delta Decoder、State Mixer、VMD auxiliary loss 能在极小数据上串通；
- smoke 可复用 Step 03 的 smoke VMD 缓存，不应要求本地先构建全量 VMD 缓存；
- 输出统一写入 `outputs/smoke/`，只检查张量形状、loss 计算、checkpoint/metrics 写入是否正常。

---

## 2. Backbone encode 接口

为了复用 CCG-xLSTM，建议给模型加一个 `encode()`：

```python
hidden_seq, feature = backbone.encode(x, batch=batch)
```

返回：

```text
hidden_seq: [B, seq_len, d_model]
feature:    [B, d_model]       # 默认取最后时间步
```

后续线性注意力也会用 `hidden_seq`。

---

## 3. VMD 多分支头

### 3.1 输入输出

输入：

```text
feature: [B, d_model]
```

输出：

```text
mode_preds_delta_or_direct: [B, pred_len, target_dim, K]
```

### 3.2 接口

`src/ship_motion/models/vmd_heads.py`：

```python
class VMDMultiHead(nn.Module):
    def __init__(self, d_model, pred_len, target_dim, K, dropout=0.1):
        pass

    def forward(self, feature):
        # return [B, pred_len, target_dim, K]
        pass
```

推荐先用一个 MLP 直接输出全部模态，避免 K 个 head 带来过多代码。

---

## 4. Delta Decoder

### 4.1 为什么要做

船舶姿态是连续变化的。直接预测未来绝对值可能出现跳变。Delta Decoder 让模型预测变化量，再从历史最后状态开始累积。

### 4.2 两种模式

配置：

```yaml
model:
  decode_type: delta   # direct | delta
```

- `direct`：直接输出未来状态；
- `delta`：输出未来增量，累积得到状态。

### 4.3 接口

`src/ship_motion/models/decoders.py`：

```python
class DeltaDecoder(nn.Module):
    def __init__(self, mode="delta"):
        pass

    def forward(self, y_base, last_state_std=None):
        """
        y_base: [B, pred_len, target_dim]
        last_state_std: [B, target_dim]
        return y_hat: [B, pred_len, target_dim]
        """
        pass
```

注意：训练输出在标准化空间中，所以 `last_state_std` 应该是标准化后的历史最后状态。可从 `x[:, -1, -5:]` 取得。

---

## 5. State Coupling Mixer

### 5.1 为什么要做

`u,v,p,r,phi` 不是互相独立的。例如：

- `p` 和 `phi` 有积分关系；
- `v` 和 `r` 在横荡/艏摇中有耦合；
- 舵角变化会同时影响多个状态。

State Mixer 用一个很小的网络在每个预测步内部修正五个状态之间的关系。

### 5.2 接口

`src/ship_motion/models/state_mixer.py`：

```python
class StateCouplingMixer(nn.Module):
    def __init__(self, target_dim=5, hidden_dim=16, dropout=0.0):
        pass

    def forward(self, y):
        """
        y: [B, pred_len, target_dim]
        return: [B, pred_len, target_dim]
        """
        pass
```

推荐实现：

```text
residual = MLP(y_t)
y_t_new = y_t + residual
```

配置：

```yaml
model:
  use_state_mixer: true
```

---

## 6. VMD-CCG-xLSTM Forecaster

```python
class VMDCCGXLSTMForecaster(nn.Module):
    def forward(self, x, batch=None):
        hidden_seq, feature = self.backbone.encode(x, batch=batch)
        mode_preds = self.vmd_head(feature)          # [B,H,5,K]
        y_base = mode_preds.sum(dim=-1)              # [B,H,5]
        y_hat = self.decoder(y_base, last_state_std)
        y_hat = self.state_mixer(y_hat) if enabled else y_hat
        return {
            "y_hat": y_hat,
            "mode_preds": mode_preds,
            "hidden_seq": hidden_seq,
        }
```

训练代码要兼容 dict output：

```python
output = model(x, batch=batch)
y_hat = output["y_hat"] if isinstance(output, dict) else output
```

---

## 7. VMD 损失

```text
pred_loss = mse(y_hat, y)
vmd_loss = mse(mode_preds, y_modes_std)
loss = pred_loss + lambda_vmd * vmd_loss
```

### 7.1 y_modes 标准化规则

推荐：

```text
y_modes_std = y_modes_raw / y_std
```

不要对每个模态减 `y_mean`。因为多个模态相加后才对应原始信号，均值不应重复减 K 次。

---

## 8. 配置文件

`configs/vmd_ccg_xlstm.yaml`：

```yaml
run_name: vmd_ccg_xlstm_seq128_pred10
data:
  train_dir: data/patrol_ship_routine/processed/train
  val_dir: data/patrol_ship_routine/processed/validation
  routine_test_dir: data/patrol_ship_routine/processed/test
  ood_test_dir: data/patrol_ship_ood/processed/test
  exog_cols: [n, deltal, deltar, Vw, alpha_x, alpha_y]
  state_cols: [u, v, p, r, phi]
  input_cols: [n, deltal, deltar, Vw, alpha_x, alpha_y, u, v, p, r, phi]
  target_cols: [u, v, p, r, phi]
  seq_len: 128
  pred_len: 10
  stride: 1
  use_vmd: true
  vmd_cache_root: outputs/cache/vmd/K3_alpha2000
model:
  name: vmd_ccg_xlstm
  exog_dim: 6
  state_dim: 5
  target_dim: 5
  pred_len: 10
  d_model: 128
  context_dim: 64
  num_layers: 2
  dropout: 0.1
  gate_clip: 5.0
  decode_type: delta
  use_state_mixer: true
vmd:
  enabled: true
  K: 3
  lambda_vmd: 0.2
train:
  seed: 42
  batch_size: 128
  epochs: 50
  lr: 0.001
  weight_decay: 0.0001
  grad_clip: 1.0
  early_stop_patience: 8
  num_workers: 0
  device: auto
```

---

## 9. 验收标准

必须满足：

1. Dataset 开启 VMD 后返回 `y_modes`；
2. 模型输出包含 `y_hat`、`mode_preds`、`hidden_seq`；
3. `mode_preds.shape == [B, pred_len, 5, K]`；
4. `y_hat.shape == [B, pred_len, 5]`；
5. Delta Decoder 可通过配置关闭；
6. State Mixer 可通过配置关闭；
7. VMD-CCG-xLSTM 能完整训练和评估。

---

## 10. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/05_vmd_ccg_integration.md 实现 VMD-CCG-xLSTM 集成。需要新增 VMDMultiHead、DeltaDecoder 和 StateCouplingMixer。模型输出 mode_preds 和 y_hat，训练损失包含预测 MSE 和 VMD 辅助 MSE。暂时不要加入线性注意力和物理约束。
```



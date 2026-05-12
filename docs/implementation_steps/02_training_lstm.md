# Step 02：常规基线模型与训练评估框架

## 这一步的目标（给小白看的说明）

这一步要完成第一套真正能训练和评估的代码。先不要急着写 xLSTM、VMD 或注意力。原因是：如果基础模型都跑不通，后面复杂模型出错时就很难判断是数据问题、训练问题还是模型问题。

本步骤要补齐常规基线，不只做 LSTM。建议实现：

1. `Persistence`：不训练，复制最后一个历史状态；
2. `LSTM`：最常见的循环网络基线；
3. `GRU`：比 LSTM 更轻量的门控循环网络；
4. `TCN`：时间卷积网络，使用因果/膨胀卷积建模时序；
5. `Transformer Encoder`：标准注意力基线，可用于对比注意力模型精度和计算代价。

这一步完成后，你应该能做到：

1. 训练 LSTM / GRU / TCN / Transformer；
2. 在 validation、routine test、OOD test 上输出 MAE、RMSE、R²；
3. 保存 checkpoint、scaler、配置文件和指标文件；
4. 确认整个“训练—验证—测试”流程是通的；
5. 为后续 xLSTM 改进模型提供可信对照组。

这一步仍然不做 VMD，也不做 xLSTM。它的作用是建立可靠实验框架和常规基线。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/
  models/
    __init__.py
    persistence.py
    lstm.py
    gru.py
    tcn.py
    transformer.py
  metrics.py
  train.py
  evaluate.py
configs/
  lstm.yaml
  gru.yaml
  tcn.yaml
  transformer.yaml
scripts/
  run_baselines.ps1
```

---

## 2. 统一模型接口

所有模型必须统一输出：

```python
y_hat = model(x, batch=None)
```

形状：

```text
x: [B, seq_len, input_dim]
y_hat: [B, pred_len, target_dim]
```

这里的 `y_hat` 是标准化空间的预测值。评估时必须反标准化后再计算指标。

---

## 3. Persistence 基线

文件：`src/ship_motion/models/persistence.py`

思路：未来 10 秒都等于历史最后一秒的状态。

```python
last_y = x[:, -1, -target_dim:]
y_hat = last_y.unsqueeze(1).repeat(1, pred_len, 1)
```

Persistence 不需要训练，但必须参与评估。它用于告诉我们：深度模型至少要比“什么都不学，只复制最后值”更好。

---

## 4. LSTM 基线

文件：`src/ship_motion/models/lstm.py`

结构：

```text
输入 x
 -> LSTM
 -> 取最后一个时间步 hidden
 -> MLP
 -> reshape 成 [B, pred_len, 5]
```

接口：

```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len, target_dim, dropout=0.1): pass
    def forward(self, x, batch=None): pass
```

推荐参数：

```yaml
hidden_dim: 128
num_layers: 2
dropout: 0.1
```

---

## 5. GRU 基线

文件：`src/ship_motion/models/gru.py`

GRU 和 LSTM 类似，但结构更轻，参数更少。它适合作为轻量 RNN 对比。

结构：

```text
输入 x
 -> GRU
 -> 取最后一个时间步 hidden
 -> MLP
 -> reshape 成 [B, pred_len, 5]
```

接口：

```python
class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, pred_len, target_dim, dropout=0.1): pass
    def forward(self, x, batch=None): pass
```

配置基本复制 LSTM，只把 `model.name` 改成 `gru`。

---

## 6. TCN 基线

文件：`src/ship_motion/models/tcn.py`

TCN 是时间卷积网络。它不按时间步递归，而是用一维卷积看历史窗口。膨胀卷积可以扩大感受野，适合时序预测。

输入输出：

```text
x: [B, L, F]
转置为 [B, F, L]
 -> 多层 causal/dilated Conv1d
 -> 取最后时间步 feature
 -> MLP
 -> [B, pred_len, 5]
```

建议模块：

```python
class TemporalBlock(nn.Module): pass
class TCNForecaster(nn.Module): pass
```

推荐参数：

```yaml
channels: [64, 128, 128]
kernel_size: 3
dropout: 0.1
```

注意：实现时要保证卷积不使用未来信息。最简单方法是用 padding 后裁剪右侧多余部分。

---

## 7. Transformer Encoder 基线

文件：`src/ship_motion/models/transformer.py`

Transformer 用标准自注意力建模历史窗口，作为注意力基线。它可能精度不错，但计算复杂度更高。

结构：

```text
Input projection
 -> Positional Encoding
 -> TransformerEncoder
 -> last token 或 mean pooling
 -> MLP head
```

接口：

```python
class TransformerForecaster(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, pred_len, target_dim, dropout=0.1): pass
    def forward(self, x, batch=None): pass
```

推荐参数：

```yaml
d_model: 128
nhead: 4
num_layers: 2
dropout: 0.1
```

如果训练时间紧张，Transformer 可以只跑 `pred_len=10` 主实验，不做多步长扩展。

---

## 8. 指标设计

文件：`src/ship_motion/metrics.py`

必须实现：

```python
def mae(y_pred, y_true): pass
def rmse(y_pred, y_true): pass
def r2_score(y_pred, y_true): pass

def compute_metrics(y_pred_raw, y_true_raw, target_cols): pass
```

输出 JSON 示例：

```json
{
  "mae_mean": 0.12,
  "rmse_mean": 0.18,
  "r2_mean": 0.91,
  "rmse_u": 0.20,
  "rmse_v": 0.10,
  "rmse_p": 0.01,
  "rmse_r": 0.02,
  "rmse_phi": 0.03
}
```

---

## 9. 训练框架

文件：`src/ship_motion/train.py`

需要实现：

```python
def build_model(config): pass

def train_one_epoch(model, loader, optimizer, device): pass

def validate(model, loader, scaler, device): pass

def fit(config): pass
```

训练损失：

```python
loss = MSELoss(y_hat_std, y_std)
```

早停指标：

```text
val_rmse_mean 越低越好
```

---

## 10. 输出目录

每次训练保存：

```text
outputs/{run_name}/
  config.yaml
  scaler.json
  best.pt
  train_log.csv
  metrics_val.json
  metrics_routine_test.json
  metrics_ood_test.json
```

---

## 11. 配置文件要求

所有 baseline 配置都使用同一数据设置：

```yaml
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

每个模型单独改 `run_name` 和 `model`：

```yaml
run_name: gru_seq128_pred10
model:
  name: gru
  input_dim: 11
  target_dim: 5
  hidden_dim: 128
  num_layers: 2
  dropout: 0.1
```

---

## 12. run_baselines.ps1

顺序执行：

```powershell
python -m ship_motion.train --config configs/lstm.yaml
python -m ship_motion.train --config configs/gru.yaml
python -m ship_motion.train --config configs/tcn.yaml
python -m ship_motion.train --config configs/transformer.yaml
```

如果训练慢，先跑 LSTM、GRU、TCN，Transformer 后补。

---

## 13. 验收标准

必须满足：

1. Persistence 可直接评估；
2. LSTM、GRU、TCN 至少能训练 1 个 epoch；
3. Transformer 能完成 shape test，最好能完整训练；
4. 指标在反标准化后计算；
5. routine test 和 OOD test 都有结果；
6. 所有 baseline 的输出格式一致，后续汇总脚本可统一读取。

---

## 14. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/02_training_lstm.md 实现常规基线模型和训练评估框架。需要实现 Persistence、LSTM、GRU、TCN、Transformer Encoder、metrics、train.py 和 evaluate.py。不要实现 VMD、xLSTM、注意力和物理约束。指标必须在反标准化后的真实物理尺度上计算。
```


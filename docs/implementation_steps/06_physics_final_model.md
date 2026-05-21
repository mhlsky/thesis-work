# Step 06：物理约束与最终主模型 VMD-CCG-Phys-xLSTM

## 这一步的目标（给小白看的说明）

这一步在 `VMD-CCG-xLSTM` 的基础上加入物理约束，形成主线最终模型：

```text
VMD-CCG-Phys-xLSTM
```

前面步骤已经解决了：

- CCG：让控制量和风场量参与门控；
- VMD：让模型按趋势、周期、扰动分量学习；
- Delta Decoder：让预测更连续；
- State Mixer：让多状态输出之间互相修正。

这一步解决的问题是：预测结果不能只追求误差低，还要符合船舶运动的基本物理规律。船舶运动有惯性，姿态不应该突然尖峰；同时 `p` 是横摇角速度，`phi` 是横摇角，两者应该近似满足积分关系。

注意：线性注意力不在本步骤实现。它已经降级为最后的可选增强，放到 `08_optional_attention.md`。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/losses/physics.py
configs/vmd_ccg_phys_xlstm.yaml
scripts/smoke_step_06_physics.ps1
scripts/train_final_model.ps1
```

修改：

```text
ship_motion/train.py              # 支持 physics loss
ship_motion/metrics.py            # 增加物理指标
```

Smoke test 要求：

- `scripts/smoke_step_06_physics.ps1` 必须验证最终主模型能在极小数据上完成前向、预测损失、VMD loss、平滑损失、横摇运动学损失和物理指标计算；
- smoke test 必须覆盖反标准化后的物理损失/物理指标路径，避免只在标准化空间验证；
- 本地 smoke test 不要求正式训练最终模型，完整训练后续放到算力平台运行。

---

## 2. 物理约束损失

物理损失必须在反标准化后的真实物理尺度上计算。

### 2.1 平滑性约束

船舶运动有惯性，未来预测序列不应有突然尖峰。使用二阶差分：

```python
d1 = y[:, 1:, :] - y[:, :-1, :]
d2 = d1[:, 1:, :] - d1[:, :-1, :]
loss_smooth = mean(d2 ** 2)
```

### 2.2 横摇运动学约束

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

### 2.3 接口

`src/ship_motion/losses/physics.py`：

```python
def smoothness_loss(y_raw): pass

def roll_kinematic_loss(y_raw, last_state_raw, dt=1.0, p_idx=2, phi_idx=4): pass

def physics_loss(y_raw, last_state_raw, lambda_smooth, lambda_roll, dt=1.0): pass
```

---

## 3. 总损失

最终主模型训练损失：

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

## 4. 配置文件

`configs/vmd_ccg_phys_xlstm.yaml`：

```yaml
run_name: vmd_ccg_phys_xlstm_seq128_pred10
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
physics:
  enabled: true
  lambda_smooth: 0.01
  lambda_roll: 0.05
  dt: 1.0
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

## 5. 物理指标

除了训练损失，还要在测试时输出物理指标：

```python
def smoothness_metric(y_pred_raw): pass

def roll_consistency_rmse(y_pred_raw, last_state_raw, dt=1.0): pass
```

论文里可以用这些指标证明：最终模型不仅误差低，而且更符合运动规律。

---

## 6. 调试顺序

不要一次把物理损失权重设太大。按顺序：

1. VMD-CCG-xLSTM 正常训练；
2. 只开 `lambda_smooth=0.001`；
3. 再开 `lambda_roll=0.01`；
4. 最后使用正式权重；
5. 如果预测误差升高明显，保留物理指标改善，并适当降低物理 loss 权重。

---

## 7. 验收标准

必须满足：

1. `VMD-CCG-Phys-xLSTM` 能训练和评估；
2. 物理损失在 raw 尺度计算；
3. 输出 routine test 和 OOD test 指标；
4. 输出 `smoothness` 和 `roll_consistency_rmse`；
5. 最终模型至少在 OOD 或物理指标上优于无物理约束版本。

---

## 8. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/06_physics_final_model.md，在已有 VMD-CCG-xLSTM 基础上加入物理约束损失，形成 VMD-CCG-Phys-xLSTM。不要实现线性注意力。物理损失必须在反标准化后的 raw 物理尺度上计算。
```


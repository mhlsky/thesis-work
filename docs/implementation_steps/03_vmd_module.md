# Step 03：VMD 分解模块与 VMD 标签缓存

## 这一步的目标（给小白看的说明）

这一步开始实现论文里的第一个必要创新点：**VMD 变分模态分解**。

船舶运动数据不是单一规律，它通常混合了低频趋势、波浪周期扰动和高频随机扰动。VMD 的作用就是把原始时序拆成几个更容易学习的分量。模型后续不用直接硬学一个复杂序列，而是分别学习几个分量，再把它们加起来。

但这里有一个非常重要的问题：**不能让模型提前看到测试集未来信息。** 所以本项目推荐把 VMD 结果作为“辅助标签”，而不是作为模型输入。也就是说，VMD 用来告诉模型“真实序列可以拆成哪些部分”，但推理时模型只用历史输入，不需要未来 VMD。

这一步完成后，你应该能做到：

1. 对每个 CSV 文件的 `u,v,p,r,phi` 做 VMD；
2. 把每个目标变量拆成 K 个模态；
3. 把分解结果缓存到 `outputs/cache/vmd/`；
4. Dataset 能在返回 `y` 的同时返回 `y_modes`；
5. 后续模型可以用 `y_modes` 计算 VMD 辅助损失。

这一步不训练最终模型，只让 VMD 数据准备好。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/data/vmd.py
ship_motion/losses/vmd_loss.py
configs/vmd_ccg_xlstm.yaml     # 可先建配置骨架，模型下一步再用
scripts/build_vmd_cache.ps1
```

修改文件：

```text
ship_motion/data/dataset.py   # 支持返回 y_modes
```

---

## 2. VMD 实现方式

### 2.1 推荐依赖

优先使用：

```text
vmdpy
```

如果后续环境不能安装，可以让 AI 实现一个 NumPy 版 VMD 或临时只跑无 VMD 流程调试。但正式论文实验必须使用 VMD。

`requirements.txt` 加入：

```text
vmdpy
```

### 2.2 VMD 参数

默认：

```yaml
vmd:
  enabled: true
  K: 3
  alpha: 2000
  tau: 0
  DC: false
  init: 1
  tol: 1e-7
  lambda_vmd: 0.2
```

解释：

- `K=3`：拆成 3 个模态，可理解为趋势、周期、扰动；
- `alpha=2000`：常用默认带宽约束；
- `lambda_vmd=0.2`：训练时 VMD 辅助损失权重。

---

## 3. 缓存设计

不要把 VMD 结果写回原始 `data/`。缓存写到：

```text
outputs/cache/vmd/K3_alpha2000/
  train/
  validation/
  routine_test/
  ood_test/
```

每个 CSV 对应一个 `.npz`：

```text
20190805-095929.npz
```

`.npz` 内容：

```python
modes: shape [T, target_dim, K]
target_cols: ["u", "v", "p", "r", "phi"]
K: 3
```

为什么是 `[T, target_dim, K]`：这样 Dataset 根据时间窗口切片最方便。

---

## 4. `vmd.py` 接口

```python
class VMDConfig:
    K: int
    alpha: float
    tau: float
    DC: bool
    init: int
    tol: float

class VMDCacheBuilder:
    def __init__(self, config, target_cols, cache_root):
        pass

    def decompose_1d(self, signal):
        # return [T, K]
        pass

    def build_file_cache(self, csv_path, split_name):
        # read csv, decompose target_cols, save npz
        pass

    def build_split_cache(self, data_dir, split_name):
        pass
```

注意：不同 VMD 库返回形状可能是 `[K, T]`，保存前要转成 `[T, K]`。

---

## 5. Dataset 接入 y_modes

`ShipWindowDataset` 增加参数：

```python
vmd_cache_dir=None
vmd_enabled=False
vmd_K=3
```

如果开启 VMD，则 `__getitem__` 额外返回：

```python
"y_modes": Tensor[pred_len, target_dim, K]
```

切片逻辑与 `y` 完全一致：

```text
y_modes = modes[start+seq_len : start+seq_len+pred_len]
```

---

## 6. VMD 辅助损失

文件：`src/ship_motion/losses/vmd_loss.py`

```python
def vmd_aux_loss(mode_preds, y_modes):
    """
    mode_preds: [B, pred_len, target_dim, K]
    y_modes:    [B, pred_len, target_dim, K]
    """
    return mse(mode_preds, y_modes)
```

最终预测后续由模型负责：

```python
y_hat = mode_preds.sum(dim=-1)
```

---

## 7. 泄漏说明

本项目主线只把 VMD 作为目标辅助标签，不把 VMD 分解结果作为模型输入。

这样做是为了避免：

```text
测试集整段分解 -> 分解结果包含未来信息 -> 模型预测虚高
```

论文里可以表述为：

> 为避免分解输入引入未来信息，本文将 VMD 模态作为辅助监督信号，引导模型学习不同频段的运动响应，推理阶段不依赖未来 VMD 分解结果。

---

## 8. 验收标准

必须满足：

1. 能为 train / validation / routine test / OOD test 生成 VMD 缓存；
2. 每个 `.npz` 中 `modes.shape == [T, 5, K]`；
3. Dataset 开启 VMD 后能返回 `y_modes`；
4. `y_modes.shape == [pred_len, 5, K]`；
5. 关闭 VMD 时，原 LSTM 训练不受影响。

---

## 9. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/03_vmd_module.md 实现 VMD 分解模块。VMD 是必做模块，但本步骤只负责生成缓存和让 Dataset 返回 y_modes，不要实现 xLSTM 模型。VMD 结果只能作为辅助标签，不作为模型输入，避免未来信息泄漏。
```



# Step 01：数据管线与 Smoke Test

## 这一步的目标（给小白看的说明）

这一步只解决一个问题：**让程序能正确读取数据，并把连续的船舶时序切成模型能吃的小窗口。**

深度学习模型不能直接把整个 CSV 文件一次性丢进去训练。我们需要把每个 1 小时 CSV 切成很多样本，例如“看过去 128 秒，预测未来 10 秒”。这一步完成后，你还不会得到预测模型，但会得到后续所有模型都依赖的数据入口。

本项目后面要做“控制/风场条件门控”，所以从数据管线开始就要明确两类字段：

```text
exog_cols  = 控制/环境量：n, deltal, deltar, Vw, alpha_x, alpha_y
state_cols = 船舶状态量：u, v, p, r, phi
```

这一步做完后，你应该能确认：

1. 程序能找到 train / validation / test / OOD 数据；
2. 每个 CSV 内部可以正确生成滑动窗口；
3. 一个样本的输入形状是 `[128, 11]`，输出形状是 `[10, 5]`；
4. batch 中同时保留 `x_exog` 和 `x_state`，后续 CCG-xLSTM 可直接使用；
5. 标准化器只在训练集上拟合，没有使用测试集信息；
6. 可以快速跑一个 smoke test，证明数据管线没问题。

这一步不要做模型、不要训练、不要 VMD。先把地基打稳。

---

## 1. 本步骤交付物

新增文件：

```text
src/
  ship_motion/
    __init__.py
    data/
      __init__.py
      dataset.py
      scaler.py
    utils.py
configs/
  base.yaml
scripts/
  smoke_step_01_data.ps1
```

---

## 2. 数据配置

`configs/base.yaml`：

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
  max_files: null
  max_windows_per_file: null
train:
  seed: 42
  batch_size: 128
  num_workers: 0
output_dir: outputs
```

为了快速调试，smoke test 可以覆盖：

```yaml
max_files: 2
max_windows_per_file: 50
```

---

## 3. StandardScaler 设计

文件：`src/ship_motion/data/scaler.py`

### 3.1 为什么要标准化

`n` 的数值可能是几百，`phi` 可能只有很小的弧度。如果不标准化，模型训练时会偏向数值大的变量，损失也不稳定。

### 3.2 接口

```python
class StandardScaler:
    def fit_csv_files(self, files, input_cols, target_cols):
        pass

    def transform_x(self, array):
        pass

    def transform_y(self, array):
        pass

    def inverse_y(self, array):
        pass

    def inverse_y_tensor(self, tensor):
        pass

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        pass
```

### 3.3 规则

- 只能用 train split 拟合均值和标准差；
- validation、routine test、OOD test 只能 transform，不能 fit；
- `std == 0` 时替换成 `1.0`；
- 保存为 JSON，后续训练和测试必须复用同一个 scaler。

---

## 4. ShipWindowDataset 设计

文件：`src/ship_motion/data/dataset.py`

### 4.1 样本切法

对于单个 CSV：

```text
x = 第 start 到 start+seq_len-1 行的 input_cols
y = 第 start+seq_len 到 start+seq_len+pred_len-1 行的 target_cols
```

示例：

```text
seq_len = 128
pred_len = 10
start = 0
x 使用第 0~127 秒
y 预测第 128~137 秒
```

### 4.2 重要限制

一个样本只能来自同一个 CSV 文件，不能把 A 文件最后几秒和 B 文件开头几秒拼成一个窗口。

### 4.3 返回格式

```python
{
    "x": Tensor[seq_len, 11],              # 标准化完整输入
    "x_exog": Tensor[seq_len, 6],          # 标准化控制/环境量
    "x_state": Tensor[seq_len, 5],         # 标准化历史状态量
    "y": Tensor[pred_len, 5],              # 标准化标签
    "y_raw": Tensor[pred_len, 5],          # 真实物理尺度标签
    "last_state_raw": Tensor[5],           # 输入窗口最后一帧状态，delta decoder 和物理损失要用
    "file": str,
    "start": int,
}
```

`x_exog` 和 `x_state` 可以由 `x` 切片得到，但建议 Dataset 直接返回，后续模型代码更清楚。

---

## 5. 工具函数

文件：`src/ship_motion/utils.py`

```python
def set_seed(seed: int): pass

def list_csv_files(data_dir): pass

def ensure_dir(path): pass

def load_yaml(path): pass

def save_json(obj, path): pass
```

---

## 6. Smoke Test

`scripts/smoke_step_01_data.ps1` 运行：

```powershell
python -m ship_motion.data.dataset --config configs/base.yaml --smoke
```

如果先不做 CLI，可以直接写一个 `run_smoke_test()` 函数。

Smoke test 要求：

- 只使用极少文件和窗口，例如 `max_files: 2`、`max_windows_per_file: 50`；
- 只验证数据发现、滑窗、标准化、batch shape 和 scaler 保存；
- 输出写入 `outputs/smoke/` 或 `outputs/smoke_test/`，不要污染正式实验目录。

### 预期输出

```text
train files: 2
val files: 2
train windows: 100
x shape: torch.Size([128, 11])
x_exog shape: torch.Size([128, 6])
x_state shape: torch.Size([128, 5])
y shape: torch.Size([10, 5])
scaler saved to outputs/smoke_test/scaler.json
```

---

## 7. 验收标准

必须满足：

1. 能读取 CSV；
2. 能构造 train / val Dataset；
3. batch shape 正确；
4. scaler 保存成功；
5. 不跨文件生成窗口；
6. 返回 `x_exog`、`x_state`、`last_state_raw`；
7. smoke test 10 秒左右能跑完。

---

## 8. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/01_data_pipeline.md 实现数据管线。只实现 StandardScaler、ShipWindowDataset、utils 和 smoke test，不要写模型、训练、VMD。窗口不能跨 CSV 文件，scaler 只能在训练集上拟合。Dataset 需要返回 x、x_exog、x_state、y、y_raw、last_state_raw。
```



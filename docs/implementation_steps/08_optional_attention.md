# Step 08（可选增强）：线性注意力增强

## 这一步的目标（给小白看的说明）

这一步是可选增强，不是主线必做。

在完成 `VMD-CCG-Phys-xLSTM` 之后，如果还有时间和算力，可以加入线性注意力，得到：

```text
VMD-CCG-Attn-Phys-xLSTM
```

线性注意力的作用是让模型在历史窗口中自动关注关键时刻，例如舵角突变、风速突变、横摇峰值等。但它不是本项目最有场景特色的创新点。真正核心的主线仍然是：

```text
VMD + CCG + Delta Decoder + State Mixer + Physics
```

如果线性注意力加入后提升不明显，可以只作为附加实验写在论文中，不必作为最终主模型。

---

## 1. 本步骤交付物

新增文件：

```text
ship_motion/models/linear_attention.py
configs/vmd_ccg_attn_phys_xlstm.yaml
scripts/smoke_step_08_attention.ps1
scripts/train_attention_enhanced.ps1
```

修改：

```text
ship_motion/models/xlstm.py 或 VMD-CCG forecaster 相关文件
ship_motion/train.py
```

Smoke test 要求：

- `scripts/smoke_step_08_attention.ps1` 必须验证线性注意力模块接入后，`hidden_seq -> attention -> VMD head -> decoder` 的路径能在极小数据上跑通；
- smoke test 应比较开启/关闭 attention 的配置是否都能完成一次前向和最小训练/评估；
- 该步骤是可选增强，本地 smoke 只用于判断集成是否正确，不要求证明精度提升。

---

## 2. 线性注意力放在哪里

放在 CCG-xLSTM 输出序列之后、VMD head 之前：

```text
Input -> CCG-xLSTM hidden_seq -> LinearSelfAttention -> feature -> VMDMultiHead -> DeltaDecoder -> StateMixer
```

这样注意力能在历史窗口中挑选关键时间片。

---

## 3. 接口设计

`src/ship_motion/models/linear_attention.py`：

```python
class LinearSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, dropout=0.1, eps=1e-6):
        pass

    def forward(self, x):
        # x: [B, L, D]
        # return: [B, L, D]
        pass
```

---

## 4. 算法说明

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

## 5. 配置文件

`configs/vmd_ccg_attn_phys_xlstm.yaml`：

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

完整配置可复制 `vmd_ccg_phys_xlstm.yaml`，只增加 `use_attention` 和 `attn_heads`。

---

## 6. 消融定位

线性注意力只做以下对比：

```text
VMD-CCG-Phys-xLSTM
vs
VMD-CCG-Attn-Phys-xLSTM
```

如果提升明显，可以在论文中作为增强模块；如果提升不明显，就作为可选实验说明：标准/线性注意力在本数据规模下收益有限，而 CCG 与物理约束贡献更明显。

---

## 7. 验收标准

必须满足：

1. `use_attention=false` 时主模型不受影响；
2. `use_attention=true` 时模型能训练和评估；
3. 注意力模块输入输出 shape 均为 `[B, seq_len, d_model]`；
4. 至少输出 routine test 和 OOD test 对比；
5. 无论是否提升，都不能影响主线 `VMD-CCG-Phys-xLSTM` 的结果。

---

## 8. 交给 AI 编码的提示词

```text
请根据 docs/implementation_steps/08_optional_attention.md，在已有 VMD-CCG-Phys-xLSTM 基础上加入可选 LinearSelfAttention。注意这是可选增强，必须通过 use_attention 开关控制。不要改变主线模型默认行为。
```


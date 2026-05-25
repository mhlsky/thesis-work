# 实验3手册：Phys / VMD 独立验证

> 适用时间：`result_2_followup` 分析之后  
> 目标结果目录：`results/result3/`  
> 运行环境：**Linux 云服务器**（先手动激活你自己的 Python / torch 环境）  
> 本手册特点：**先不改核心代码**，只通过新的运行时配置与批量脚本完成实验3。

---

## 1. 为什么要做实验3

经过 `result_1` 和 `result_2_followup` 两轮结果，现在已经比较明确：

1. `CCG-xLSTM` 仍然是当前最稳的强基线；
2. `VMD-CCG-xLSTM` 在 `result_2_followup` 中虽然比 `result_1` 明显回升，但仍未反超 `CCG-xLSTM`；
3. `VMD-CCG-Phys-xLSTM` 与 `VMD-CCG-xLSTM` 结果几乎重合，说明 **physics 分支的独立收益还没有被清楚验证**；
4. `result_2_followup` 中 `best_epoch=1`，说明当前 VMD / physics 辅助信号可能仍然存在“最终强度偏大”的问题。

因此，实验3不再追求“直接堆最终主模型”，而是先回答两类更基础的问题：

- **Phys 单独有没有价值？**
- **VMD 单独为什么还不稳？**

---

## 2. 实验3的设计原则

实验3遵守下面 6 条原则：

1. **Phys 与 VMD 分开验证**
   - 不再把两者绑在同一个最终模型里解释；
2. **不改数据划分**
   - 继续沿用当前 train / validation / routine_test / ood_test；
3. **尽量不改 backbone**
   - 统一以 `CCG-xLSTM` 为主干；
4. **先做配置层实验，不改训练代码**
   - 当前实验3只动 YAML 运行配置；
5. **统一写到 `results/result3/`**
   - 方便和 `result_1`、`result_2_followup` 对照；
6. **重点看 OOD、`u/v` 和物理指标**
   - 不只盯 `rmse_mean`。

---

## 3. 实验3的两大实验组

实验3分成两组：

1. **Phys 独立验证组**
2. **VMD 独立验证组**

其中：

- Phys 组重点回答：  
  **physics 本身有没有帮助，哪一类物理约束更有效？**
- VMD 组重点回答：  
  **VMD 为什么仍然不稳，究竟是 warmup、学习率、分解参数，还是后耦合结构的问题？**

---

## 4. Phys 独立验证组

## 4.1 设计思路

Phys 组不再挂在 `VMD-CCG-xLSTM` 后面，而是直接挂在当前最稳的：

- `CCG-xLSTM`

这样能干净回答：

> **不加 VMD，只加 physics，到底有没有帮助？**

## 4.2 具体实验

### P0. 基线对照

- run_name: `e3_ccg_xlstm_base`
- 目的：
  - 作为本轮 Linux 云服务器环境下的统一对照；
  - 避免只拿旧实验跨轮对比。

### P1. 只开 smoothness

- run_name: `e3_ccg_phys_s001_r000`
- 主要配置：
  - `lambda_smooth = 0.001`
  - `lambda_roll = 0`
- 目的：
  - 验证平滑项本身有没有正向作用；
  - 看它是否会伤害 `u / v`。

### P2. 只开 roll consistency

- run_name: `e3_ccg_phys_s000_r005`
- 主要配置：
  - `lambda_smooth = 0`
  - `lambda_roll = 0.005`
- 目的：
  - 单独验证横摇一致性项；
  - 看它是否能在不明显伤主误差的前提下改善 physics 指标。

### P3. 保守组合版

- run_name: `e3_ccg_phys_s001_r005`
- 主要配置：
  - `lambda_smooth = 0.001`
  - `lambda_roll = 0.005`
- 目的：
  - 检查两个物理项一起上时是否能比单项更稳。

### P4. 标准组合版

- run_name: `e3_ccg_phys_s001_r010`
- 主要配置：
  - `lambda_smooth = 0.001`
  - `lambda_roll = 0.01`
- 目的：
  - 在比 P3 略强的 roll 惩罚下，再看是否有更多收益；
  - 同时观察是否又出现类似 `result_2_followup` 那样“约束一强就变差”的迹象。

## 4.3 Phys 组的统一设置

所有 Phys 组实验统一使用：

- `warmup_epochs = 8`
- `smoothness_target_cols = [p, r, phi]`
- `normalize_by_target_std = true`
- `roll_integration = trapezoid`
- `p_col = p`
- `phi_col = phi`

这样做是为了：

- 避免平滑项直接粗暴作用到 `u / v`
- 保持 physics 指标口径一致
- 让 Phys 组内部可以直接横向比较

---

## 5. VMD 独立验证组

## 5.1 设计思路

VMD 组继续用：

- `CCG-xLSTM + VMD`

但这次不再只问“降权重有没有用”，而是把 VMD 的问题拆成 4 个方向：

1. **warmup 节奏**
2. **学习率**
3. **VMD 分解参数（K / alpha）**
4. **VMD 后耦合结构（state mixer）**

另外，为了给这 4 个方向一个共同参考点，这一组会先跑一个低权重基线：

- `lambda_vmd = 0.05`

## 5.2 具体实验

### V0. 低权重公共基线

- run_name: `e3_vmd_ccg_l005_base`
- 主要配置：
  - `lambda_vmd = 0.05`
  - `warmup_epochs = 5`
  - `lr = 0.001`
  - `K = 3`
  - `alpha = 2000`
  - `use_state_mixer = true`

目的：

- 作为实验3里所有 VMD 方向的统一参考点；
- 回答“当最终权重直接降到较低水平后，VMD 是否已经足够稳定”。

### V1. warmup 方向

- run_name: `e3_vmd_ccg_l005_warm10`
- 主要配置：
  - `lambda_vmd = 0.05`
  - `warmup_epochs = 10`

目的：

- 验证问题是不是“辅助监督介入得太快”；
- 看更慢的 warmup 是否会让后续 epoch 更稳。

### V2. 学习率方向

- run_name: `e3_vmd_ccg_l005_lr5e4`
- 主要配置：
  - `lambda_vmd = 0.05`
  - `lr = 5e-4`

目的：

- 验证是不是 VMD 分支让优化景观更复杂，导致原始学习率偏大；
- 看较小学习率能否避免“第 1 轮最佳、后续越训越差”。

### V3. 分解参数方向（方案 A）

- run_name: `e3_vmd_ccg_l005_k2_a1000`
- 主要配置：
  - `K = 2`
  - `alpha = 1000`

目的：

- 验证当前 `K=3, alpha=2000` 是否把动态拆得过细 / 过硬；
- 用更少模态、更弱带宽约束观察辅助标签是否更稳。

### V4. 分解参数方向（方案 B）

- run_name: `e3_vmd_ccg_l005_k4_a3000`
- 主要配置：
  - `K = 4`
  - `alpha = 3000`

目的：

- 从相反方向测试：更多模态、更强带宽约束会不会更好；
- 和 V3 一起构成一组“偏粗 / 偏细”的分解对照。

### V5. 去掉 state mixer

- run_name: `e3_vmd_ccg_l005_nomixer`
- 主要配置：
  - `use_state_mixer = false`

目的：

- 验证问题是不是不在 VMD 本身，而在 **VMD + state mixer 的耦合**；
- 如果去掉 mixer 后结果更稳，说明后耦合结构值得单独消融。

---

## 6. 实验3的完整实验清单

| 顺序 | run_name | 分组 | 核心目的 |
|---|---|---|---|
| 0 | `e3_ccg_xlstm_base` | 基线 | 本轮统一对照 |
| 1 | `e3_ccg_phys_s001_r000` | Phys | 只验证 smoothness |
| 2 | `e3_ccg_phys_s000_r005` | Phys | 只验证 roll consistency |
| 3 | `e3_ccg_phys_s001_r005` | Phys | 保守组合版 |
| 4 | `e3_ccg_phys_s001_r010` | Phys | 略强组合版 |
| 5 | `e3_vmd_ccg_l005_base` | VMD | 低权重共同基线 |
| 6 | `e3_vmd_ccg_l005_warm10` | VMD | warmup 方向 |
| 7 | `e3_vmd_ccg_l005_lr5e4` | VMD | 学习率方向 |
| 8 | `e3_vmd_ccg_l005_k2_a1000` | VMD | 分解参数方向 A |
| 9 | `e3_vmd_ccg_l005_k4_a3000` | VMD | 分解参数方向 B |
| 10 | `e3_vmd_ccg_l005_nomixer` | VMD | 去 mixer 方向 |

---

## 7. 推荐执行顺序

如果算力允许，推荐直接跑全套：

```text
P0 -> P1 -> P2 -> P3 -> P4 -> V0 -> V1 -> V2 -> V3 -> V4 -> V5
```

如果算力紧张，建议按两阶段跑：

### 第一阶段

先跑：

- `e3_ccg_xlstm_base`
- `e3_ccg_phys_s001_r000`
- `e3_ccg_phys_s000_r005`
- `e3_ccg_phys_s001_r005`
- `e3_vmd_ccg_l005_base`
- `e3_vmd_ccg_l005_warm10`
- `e3_vmd_ccg_l005_lr5e4`

### 第二阶段

再补：

- `e3_ccg_phys_s001_r010`
- `e3_vmd_ccg_l005_k2_a1000`
- `e3_vmd_ccg_l005_k4_a3000`
- `e3_vmd_ccg_l005_nomixer`

---

## 8. Linux 云服务器运行方式

## 8.1 先激活环境

例如：

```bash
conda activate your_env
```

要求这个环境至少已有：

- `torch`
- `numpy`
- `scikit-learn`
- `pyyaml`
- `vmdpy`

## 8.2 跑完整实验3

在项目根目录：

```bash
bash scripts/run_result3_experiments.sh . result3
```

## 8.3 只跑 Phys 组

```bash
RUN_GROUP=phys bash scripts/run_result3_experiments.sh . result3
```

## 8.4 只跑 VMD 组

```bash
RUN_GROUP=vmd bash scripts/run_result3_experiments.sh . result3
```

## 8.5 只跑某几个实验

例如只跑：

- `e3_ccg_xlstm_base`
- `e3_vmd_ccg_l005_base`
- `e3_vmd_ccg_l005_lr5e4`

可以用：

```bash
RUN_ONLY=e3_ccg_xlstm_base,e3_vmd_ccg_l005_base,e3_vmd_ccg_l005_lr5e4 \
bash scripts/run_result3_experiments.sh . result3
```

---

## 9. 输出目录说明

实验3的正式结果统一写到：

```text
results/result3/
  logs/
  outputs/
    e3_ccg_xlstm_base/
    e3_ccg_phys_s001_r000/
    ...
    e3_vmd_ccg_l005_nomixer/
    summary/
```

额外说明：

- 运行时 YAML 会临时生成到：
  - `runtime_configs_result3/`
- 这是为了绕开当前 `default_run_dir()` 的路径双层嵌套问题；
- 也就是说，实验3脚本已经在**不改核心代码**的前提下规避了这个已知 bug。

---

## 10. 结果分析时优先看什么

## 10.1 Phys 组

先看：

1. `OOD rmse_mean`
2. `OOD rmse_u`
3. `OOD rmse_v`
4. `roll_consistency_rmse`
5. `smoothness`

你最想回答的是：

- 只开 smoothness 是否就会伤主误差？
- 只开 roll consistency 是否更稳？
- 组合版是否真的优于单项？

## 10.2 VMD 组

先看：

1. `OOD rmse_mean`
2. `OOD rmse_u`
3. `OOD rmse_v`
4. `best_epoch`
5. `train_log.csv` 里的验证曲线

你最想回答的是：

- 更慢 warmup 是否让最优点不再卡在第 1 轮？
- 更小 lr 是否更稳？
- `K/alpha` 是否比单纯调 loss 更关键？
- 去掉 mixer 是否能说明问题在后耦合结构？

---

## 11. 实验3的成功标准

实验3不要求一步就超过所有基线，更现实的成功标准是：

### Phys 成功

满足任一条即可认为值得继续：

1. `ccg_phys_*` 在 OOD 上优于 `e3_ccg_xlstm_base`
2. 主误差几乎不变，但 physics 指标明显更好

### VMD 成功

满足任一条即可认为值得继续：

1. 某个 VMD 方向明显优于 `e3_vmd_ccg_l005_base`
2. 且和 `CCG baseline` 的差距显著缩小
3. 同时 `best_epoch` 不再死在第 1 轮

---

## 12. 当前这份实验3手册不包含什么

为了遵守“先不改核心代码”的前提，这份实验3暂时**不包含**：

1. 按变量选择性施加 VMD 监督
2. 新增 `phys_xlstm` 模型类
3. 改数据集返回字段
4. 改训练/评估主逻辑

如果实验3结束后仍然看不出清楚趋势，下一轮才建议考虑：

- VMD 按目标维度加权
- 物理约束更细粒度设计
- 单独实现新的结构版本

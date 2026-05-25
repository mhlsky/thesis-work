# result_1 后第一次反馈微调记录

> 记录时间：2026-05-25  
> 对应上一轮结果：`docs/result_analysis_result_1.md`  
> 目的：把 **第一次正式实验（result_1）分析后，已经落地到代码 / 配置 / 脚本中的调整** 记录清楚，方便后续继续对比 `result_2_followup` 或你之后的新结果。  
> 说明：当前环境里 `git` 命令不在 PATH，我是通过 `.git/logs/HEAD`、`.git/logs/refs/heads/master` 与当前仓库文件内容交叉整理这份记录的。以下内容以**当前仓库已落地实现**为准。

---

## 1. 第一次实验（result_1）暴露出来的核心问题

根据 `docs/result_analysis_result_1.md`，第一次正式实验后的主要结论是：

1. **流程问题**：`persistence_seq128_pred10` 缺正式输出，导致汇总脚本直接报错退出；
2. **有效结论**：`CCG-xLSTM` 在 OOD 上是当前最稳、最有价值的正结果；
3. **主要退化点**：
   - `VMD-CCG-xLSTM` 相比 `CCG-xLSTM` 明显退化；
   - `VMD-CCG-Phys-xLSTM` 没有把这部分退化拉回来；
4. **重点怀疑方向**：
   - VMD 辅助监督在训练初期干扰主任务；
   - physics 约束加得过早、过硬；
   - 平滑项可能误伤 `u / v` 这类主运动变量；
   - 物理指标计算虽然有意义，但约束形式还不够稳。

---

## 2. 第一次反馈后，已经落地的调整总览

## 2.1 没有动的部分

第一次反馈后，**没有直接重写主干模型结构**，也没有推翻主线：

- `Lite-xLSTM` 主干没改；
- `CCG-xLSTM` 主干没改；
- `VMD-CCG-xLSTM` 的主体框架（CCG backbone + VMD head + delta decoder + state mixer）没改；
- `VMD-CCG-Phys-xLSTM` 仍然是在 Step 05 基础上叠加 physics loss。

也就是说，这一轮不是“换模型”，而是**围绕训练稳定性、约束施加方式、结果汇总流程**做第一次反馈微调。

## 2.2 已落地的调整方向

已经落地的调整主要有 4 类：

1. **汇总流程修复**：缺 run 时不再整次崩掉；
2. **VMD 训练更保守**：引入 VMD loss warmup，而不是一开始就满权重介入；
3. **physics 约束更稳**：引入 physics warmup、按变量筛选平滑项、按尺度归一化、横摇一致性改成更稳的积分方式；
4. **补充实验组织方式明确化**：新增专门的 follow-up 正式实验脚本，只重跑真正受改动影响的模型。

---

## 3. 代码层面的实际调整

## 3.1 汇总脚本改为“缺失可跳过”

### 对应文件

- `src/ship_motion/summarize_results.py`

### 落地内容

第一次实验里，`Persistence` 正式结果缺失会导致：

- `outputs/summary/` 无法生成；
- 主表分析被流程错误卡住。

现在已经改成：

1. 某个 `run_dir` 不存在时，默认**警告并跳过**；
2. `metrics_routine_test.json` 或 `metrics_ood_test.json` 缺失时，也默认**警告并跳过**；
3. 保留 `--strict-missing-runs` 开关，如果以后要做严格完整性检查，仍可恢复“缺失即失败”。

### 这次调整解决了什么

- 以后单个 baseline 缺失时，不会阻断整个 result 的主表汇总；
- 更适合先分析已有结果，再决定补跑哪些实验。

---

## 3.2 VMD 辅助损失加入 warmup

### 对应文件

- `src/ship_motion/train.py`
- `configs/vmd_ccg_xlstm.yaml`
- `configs/vmd_ccg_phys_xlstm.yaml`

### 落地内容

训练入口里新增了：

- `resolve_linear_warmup_scale(...)`
- `build_effective_lambda_vmd(...)`

当前配置中已经加入：

- `vmd.warmup_epochs: 5`

实际训练时不再直接使用固定 `lambda_vmd`，而是按 epoch 线性放大：

- 前几轮先以较小有效权重训练；
- 到 warmup 结束后，再逐步接近配置里的目标权重。

### 这次调整背后的意图

第一次实验里最明显的问题之一是：

> 一加 VMD，主误差就明显退化。

所以这轮没有先粗暴改掉整个 VMD 结构，而是先尝试：

- **让主预测分支先站稳**；
- **再把 VMD 辅助监督慢慢拉起来**；
- 观察退化是否来自“辅助任务介入过早”。

### 一个需要明确说明的点

当前基础配置里的：

- `lambda_vmd` 仍然是 `0.2`

也就是说，这一轮落地的不是“直接把 `0.2` 改成 `0.05`”，而是：

- **保留原目标权重**
- **先通过 warmup 降低训练初期干扰**

这点后面分析 `result_2_followup` 时要特别注意，不要误以为这轮已经把 base weight 改小了。

---

## 3.3 physics 约束从“直接施加”改为“更温和、更有针对性地施加”

### 对应文件

- `src/ship_motion/train.py`
- `src/ship_motion/evaluate.py`
- `src/ship_motion/losses/physics.py`
- `configs/vmd_ccg_phys_xlstm.yaml`

### 已落地的 5 个关键细化

#### (1) physics loss 也加入了 warmup

当前配置里已经加入：

- `physics.warmup_epochs: 8`

训练时会先构造 `build_effective_physics_cfg(...)`，把：

- `lambda_smooth`
- `lambda_roll`

按 epoch 线性放大，而不是从第 1 轮就满强度开启。

#### (2) 平滑约束不再默认无差别作用到全部目标

当前配置中：

- `smoothness_target_cols: [p, r, phi]`

这表示平滑项优先约束：

- 角速度 `p`
- 航向角速度 `r`
- 横摇角 `phi`

而**不优先直接抹平 `u / v`**。

这正对应第一次实验里的一个担忧：

> OOD 上退化最明显的是 `u / v`，所以平滑项不能太粗暴。

#### (3) 物理损失可以按目标标准差做尺度归一化

当前配置中：

- `normalize_by_target_std: true`

评估/训练时会解析 `y_std`，把 physics loss 放到更可比的尺度上，避免：

- 某个物理量数值范围更大；
- 导致它在约束中天然占主导。

#### (4) 横摇一致性从单边欧拉扩展为更稳的梯形积分

当前配置中：

- `roll_integration: trapezoid`

`roll_kinematic_loss(...)` 现在支持：

- `euler`
- `trapezoid`

当前默认使用更稳的 `trapezoid`，用于离散时间下的 `p -> phi` 一致性约束。

#### (5) physics 相关目标维优先按列名解析

当前配置中：

- `p_col: p`
- `phi_col: phi`

评估与训练共用 `resolve_physics_runtime_options(...)` 来解析：

- `p_idx`
- `phi_idx`
- `smoothness_target_indices`

这样做的好处是：

- 即使以后目标列顺序调整，也不容易把 physics 约束绑错变量。

---

## 3.4 physics 指标与训练约束的解析方式统一

### 对应文件

- `src/ship_motion/evaluate.py`

### 落地内容

现在评估时，如果 `physics.enabled=true`，会用和训练阶段一致的运行时解析逻辑，统一计算：

- `smoothness`
- `roll_consistency_rmse`

这意味着：

- 训练里约束谁；
- 评估里怎么看 physics 指标；

两边已经更一致了，后续分析不会再出现“训练和评估看的不是同一套 physics 定义”的歧义。

---

## 4. 实验组织和脚本层面的调整

## 4.1 新增第一次实验后的专用补充实验脚本

### 对应文件

- `scripts/run_followup_experiments.ps1`
- `scripts/run_followup_experiments.sh`
- `docs/experiment_manual.md`

### 落地内容

已经新增“第一次正式实验后的补充实验”脚本，默认只跑：

1. `persistence`
2. `vmd_ccg_xlstm`
3. `vmd_ccg_phys_xlstm`

可选再补：

4. `ccg_xlstm` 复现实验

### 为什么这样设计

因为第一次反馈后，真正受改动影响的主要是：

- 汇总流程；
- VMD 训练逻辑；
- physics 训练/评估逻辑；
- 缺失的 persistence baseline。

而下面这些模型主干和训练目标没有本轮直接改动，所以默认**不重复训练**：

- `lstm`
- `gru`
- `transformer`
- `lite_xlstm`
- `ccg_xlstm`

这样能把算力优先花在最关键的验证点上。

---

## 4.2 正式补充实验统一写入新的 result 目录

### 对应文件

- `src/ship_motion/prepare_formal_result.py`
- `scripts/run_followup_experiments.ps1`

### 落地内容

当前 follow-up 流程会把运行时配置统一改写到：

- `results/<result_name>/runtime_configs/`

并把产物统一写到：

- `results/<result_name>/outputs/`
- `results/<result_name>/logs/`

### 这次调整解决了什么

- 避免新结果覆盖 `result_1`；
- 方便后续直接做 `result_1` vs `result_2_followup` 对照；
- 有利于后面继续加 `result_3`、`result_4` 这种滚动实验记录。

---

## 5. 从“第一次反馈建议”到“当前实际落地”的对应关系

| 第一次反馈建议 | 当前是否已落地 | 实际落地方式 |
|---|---|---|
| 补跑 Persistence | 已落地到脚本 | `run_followup_experiments.*` 默认包含 persistence |
| 缺 run 时不要整次汇总崩掉 | 已落地 | `summarize_results.py` 支持跳过缺失 run |
| VMD 不要一开始就强监督 | 已落地 | `vmd.warmup_epochs: 5` + `build_effective_lambda_vmd()` |
| physics 不要一开始就重惩罚 | 已落地 | `physics.warmup_epochs: 8` + `build_effective_physics_cfg()` |
| 重点关注 `u / v`，不要被平滑项误伤 | 部分落地 | 平滑项先收缩到 `[p, r, phi]` |
| physics 约束要更稳 | 已落地 | 标准差归一化 + 梯形积分 + 列名解析 |
| 直接把 `lambda_vmd` 降到 `0.05` | **尚未直接落地** | 当前先保留 base weight=0.2，改为 warmup 策略 |
| 直接把 `lambda_smooth/lambda_roll` 改成更小常数 | **尚未直接落地** | 当前先保留 base weight=0.01/0.05，改为 warmup + 约束细化 |

---

## 6. 这一轮微调的真实含义

如果用一句话总结第一次反馈后的落地动作，可以写成：

> **没有直接换模型，而是先把“VMD/physics 约束施加得过猛、过早、过粗”的问题，改成“更晚介入、更有选择、更按尺度归一”的版本，再做补充实验验证。**

也就是说，当前 follow-up 更像是在回答下面两个问题：

1. `result_1` 的退化，是否主要来自**训练方式太激进**？
2. 如果先不改大结构，只把约束施加得更稳，`VMD-CCG-xLSTM` 和 `VMD-CCG-Phys-xLSTM` 能不能明显回升？

---

## 7. 后续你把新结果给我时，我会优先看什么

后面你把修改后的实验结果发给我时，我会优先按下面顺序分析：

1. **Persistence 是否补齐**
   - 主表是否完整；
   - summary 是否正常生成。

2. **VMD warmup 后是否止住主误差退化**
   - 重点比较：
     - `ccg_xlstm`
     - `vmd_ccg_xlstm`

3. **physics 微调后是否在“不明显伤主误差”的前提下提升物理指标**
   - 重点比较：
     - `vmd_ccg_xlstm`
     - `vmd_ccg_phys_xlstm`

4. **OOD 上的 `u / v` 是否回升**
   - 这是第一次实验里最关键的退化观察点。

5. **回升是来自 warmup，还是来自 physics 细化**
   - 如果 Step 05 已明显回升，而 Step 06 仍不稳，说明问题主要还在 physics；
   - 如果 Step 05 就仍然明显退化，说明 VMD 主线本身还要继续改。

---

## 8. 给下一次结果分析时的备注

后续如果你给我新的 `results/<result_name>/logs/` 和 `results/<result_name>/outputs/`，我会默认：

- 它是基于这份文档记录的“第一次反馈微调版”代码跑出来的；
- 对照基线优先使用 `result_1`；
- 核心判断标准是：
  1. 主误差是否回升；
  2. OOD 是否改善；
  3. `u / v` 是否不再被明显伤害；
  4. physics 指标提升是否开始变得“值得”。

如果你后面又继续改了：

- `lambda_vmd`
- `lambda_smooth`
- `lambda_roll`
- 模型结构
- 数据划分

记得一起告诉我，我会把它视为**第二轮反馈微调**，不和这份文档混在一起分析。

---

## 9. `result_2_followup` 对比 `result_1` 的结果分析

> 分析时间：2026-05-25  
> 对比对象：  
> - `outputs/results/result_1/outputs/`  
> - `outputs/results/result_2_followup/results/result_2_followup/outputs/`  
>
> 说明：这次 follow-up 默认只补跑了：
>
> - `persistence`
> - `vmd_ccg_xlstm`
> - `vmd_ccg_phys_xlstm`

## 9.1 先说结论

这次 follow-up 可以概括成 4 句话：

1. **第一次反馈微调是有效的**：`VMD-CCG-xLSTM` 和 `VMD-CCG-Phys-xLSTM` 相比 `result_1` 都明显回升；
2. **但还不够**：两条 VMD 主线虽然不再“严重退化”，但仍然**打不过 `CCG-xLSTM` 和 `Transformer`**；
3. **physics 分支目前几乎没有额外收益**：`VMD-CCG-Phys-xLSTM` 与 `VMD-CCG-xLSTM` 的主指标几乎重合；
4. **训练信号暴露了新问题**：两条 follow-up 主线的 `best_epoch` 都是 **第 1 轮**，说明 warmup 虽然缓解了训练初期问题，但**后续把辅助/physics 权重继续拉高后，模型又开始变差**。

---

## 9.2 这次补充实验补齐了什么

和 `result_1` 相比，这次至少补齐了两个很重要的缺口：

### (1) Persistence 正式结果补齐了

现在已经有：

- `persistence_seq128_pred10/metrics_val.json`
- `persistence_seq128_pred10/metrics_routine_test.json`
- `persistence_seq128_pred10/metrics_ood_test.json`

说明后续主表 baseline 不再缺这一项。

### (2) VMD / physics 微调后的正式结果已经有了

现在可以直接对比：

- `result_1` 原始 VMD 主线
- `result_2_followup` 的 warmup / physics 细化版本

这让“第一次反馈微调到底有没有用”可以被正式回答。

---

## 9.3 和 `result_1` 的主指标对比

## 9.3.1 Routine Test

| 模型 | result_1 RMSE_mean | result_2_followup RMSE_mean | 变化 |
|---|---:|---:|---:|
| VMD-CCG-xLSTM | 0.048171 | 0.040185 | **-16.58%** |
| VMD-CCG-Phys-xLSTM | 0.046727 | 0.040216 | **-13.93%** |

补充观察：

- `VMD-CCG-xLSTM` 的 `rmse_v` 从 `0.059190 -> 0.027945`，下降约 **52.79%**；
- `VMD-CCG-Phys-xLSTM` 的 `rmse_v` 从 `0.057414 -> 0.027874`，下降约 **51.45%**；
- 说明这次回升最明显的一个点就是：**第一次实验里被明显拉坏的 `v`，确实被救回来了**。

但同时也要看到：

- `CCG-xLSTM (result_1)` 的 routine `rmse_mean = 0.030964`
- `Transformer (result_1)` 的 routine `rmse_mean = 0.030851`

因此当前 follow-up 的 VMD 两条线在 routine 上仍然：

- 比 `CCG-xLSTM` 高约 **29.8%**
- 比 `Transformer` 高约 **30.3%**

也就是说：

> **routine 上，VMD 主线虽然回升了很多，但距离“可作为主模型替代 CCG / Transformer”还差一截。**

## 9.3.2 OOD Test

| 模型 | result_1 RMSE_mean | result_2_followup RMSE_mean | 变化 |
|---|---:|---:|---:|
| VMD-CCG-xLSTM | 0.234516 | 0.195872 | **-16.48%** |
| VMD-CCG-Phys-xLSTM | 0.235505 | 0.195851 | **-16.84%** |

补充观察：

- `VMD-CCG-xLSTM` 的 `rmse_u` 从 `0.443725 -> 0.350312`，下降约 **21.05%**；
- `VMD-CCG-Phys-xLSTM` 的 `rmse_u` 从 `0.445235 -> 0.350255`，下降约 **21.33%**；
- 说明第一次实验里 OOD 上最严重的 `u` 退化，也确实被明显缓解了。

但当前最强参照仍然是：

- `CCG-xLSTM (result_1)`：`0.182219`
- `Transformer (result_1)`：`0.185287`

而 follow-up 两条 VMD 线仍然：

- 比 `CCG-xLSTM` 高约 **7.49%**
- 比 `Transformer` 高约 **5.70%**

所以现在可以更准确地说：

> **第一次反馈微调解决了“VMD 一加就大退化”的问题，但还没有把 VMD 主线推到优于当前最佳基线的程度。**

---

## 9.4 physics 分支这次有没有额外帮助

从主误差看，`VMD-CCG-Phys-xLSTM` 与 `VMD-CCG-xLSTM` 几乎重合：

### Routine

- VMD：`0.040185`
- VMD + Phys：`0.040216`

### OOD

- VMD：`0.195872`
- VMD + Phys：`0.195851`

这意味着：

- routine 上 physics 反而略差；
- OOD 上 physics 只好了一点点；
- 这个差距小到**暂时不能当作“physics 分支有效”的强证据**。

所以本轮更像是：

- **VMD warmup 起了主要作用**
- physics 细化目前**还没有独立贡献出可感知的主指标优势**

---

## 9.5 这次 physics 指标不能直接和 `result_1` 做绝对值对比

虽然 `result_2_followup` 里 physics 指标看起来大幅下降，例如：

- `smoothness`
- `roll_consistency_rmse`

但是这里有一个很重要的口径变化：

### `result_1` 的 physics 配置较旧

`result_1` 的 `config.yaml` 中只有：

- `lambda_smooth`
- `lambda_roll`

没有下面这些细化项：

- `warmup_epochs`
- `smoothness_target_cols`
- `normalize_by_target_std`
- `roll_integration: trapezoid`
- `p_col / phi_col`

### `result_2_followup` 的 physics 配置较新

新增了：

- 只对 `[p, r, phi]` 统计/约束平滑性；
- `roll_consistency_rmse` 按 `trapezoid` 口径评估；
- 训练里还使用了尺度归一化。

因此：

> **`result_1` 和 `result_2_followup` 的 physics 指标不是严格同口径，不能直接把数值下降解释成“物理一致性真实提升了这么多”。**

如果后面要严谨写论文，最好做一件事：

- 用**当前新的 physics 指标口径**，把 `result_1` 的预测结果重新离线重算一次；
- 然后再和 `result_2_followup` 比。

---

## 9.6 训练日志暴露出的新问题

这次比单看最终 RMSE 更值得注意的是训练过程本身。

### 两条 follow-up 主线的共同现象

- `best_epoch = 1`
- 后续 epoch 的 `val_rmse_mean` 都比第 1 轮更差

例如：

- `VMD-CCG-xLSTM`：
  - epoch 1: `val_rmse_mean = 0.040403`
  - epoch 2: `0.066688`
  - epoch 3: `0.092515`
- `VMD-CCG-Phys-xLSTM`：
  - epoch 1: `val_rmse_mean = 0.040401`
  - epoch 2: `0.066492`
  - epoch 3: `0.067154`

而日志里第 1 轮起始的有效权重是：

- `lambda_vmd_eff = 0.0400`
- `lambda_smooth_eff = 0.0013`
- `lambda_roll_eff = 0.0063`

这给出一个非常强的信号：

> **当前最好的表现，恰好出现在“辅助损失 / physics 约束还比较轻”的阶段。**

换句话说，warmup 的确救了训练初期，但当 warmup 继续把辅助权重推高后，模型又开始偏离最优点。

这说明现在的主要问题已经从：

- “一开始就加太猛”

变成了：

- “最终目标权重本身可能还是偏大”

---

## 9.7 这次还暴露了一个流程问题：follow-up 输出路径发生了双层嵌套

这次结果实际落在：

- `outputs/results/result_2_followup/results/result_2_followup/outputs/...`

而不是更直观的：

- `outputs/results/result_2_followup/outputs/...`

从当前代码看，问题来源很明确：

### 当前 `default_run_dir()` 的路径推导方式

它会把：

- `config["__config_path__"]` 的上两级目录

当成 `repo_root`。

但 follow-up 的运行时配置本身放在：

- `.../result_2_followup/runtime_configs/*.yaml`

这会导致路径解析时把：

- `result_2_followup/`

错误地当成“仓库根目录”，于是相对 `output_dir=results/result_2_followup/outputs` 又被再拼一次，最终变成双层：

- `result_2_followup/results/result_2_followup/outputs`

### 这会带来的实际影响

1. 结果目录结构不直观；
2. 后续 summary / plotting / 分析脚本更容易读错路径；
3. 如果后面再套一层脚本，很容易继续把路径问题放大。

这是一个应该尽快修的**工程性 bug**。

---

## 9.8 当前阶段的综合判断

如果把 `result_1 -> result_2_followup` 看成一次完整反馈闭环，那么现在的结论应该是：

### 已经验证成功的部分

1. **第一次反馈微调方向是对的**
   - warmup + 更稳的 physics 配置，确实显著缓解了 VMD 主线的退化；
2. **最严重的 `u / v` / `v` 问题得到缓解**
   - 尤其是 `v` 的 routine 误差和 `u` 的 OOD 误差回升明显；
3. **Persistence 正式结果已补齐**
   - baseline 表可以完整了。

### 仍未验证成功的部分

1. `VMD-CCG-xLSTM` 仍然不是当前最佳模型；
2. `VMD-CCG-Phys-xLSTM` 仍然没有体现出可明确归因的 physics 增益；
3. 当前最优点仍出现在很早的 epoch，说明辅助损失最终强度仍可能偏大；
4. physics 指标新旧口径不一致，暂时不适合直接拿来写“物理一致性显著提升”的结论。

因此，当前最稳的学术表述仍然是：

> **第一次反馈微调证明了 VMD/physics 主线的退化并非不可救，但现阶段它们仍然没有超过 CCG-xLSTM 这一条最稳主线。**

---

## 9.9 后续改进建议

建议按优先级分成 4 组。

### P0：先修路径与结果组织问题

1. 修 `default_run_dir()` / 运行时配置的 repo root 解析逻辑；
2. 保证正式结果统一落在单层：
   - `results/<result_name>/outputs/...`
3. follow-up 脚本结束后可选自动执行一次：
   - summary
   - plot

否则每次人工分析都要自己绕过双层目录。

### P1：下一轮先直接降最终辅助权重，而不是只靠 warmup

从这次 `best_epoch=1` 看，下一轮更应该测试：

#### VMD 分支

- 方案 A：`lambda_vmd = 0.05`，保留 warmup
- 方案 B：`lambda_vmd = 0.10`，保留 warmup
- 方案 C：`lambda_vmd = 0.05`，缩短 warmup 到 3 epoch

目标是验证：

- 是不是**最终目标权重过大**，而不是单纯“起步太猛”。

### P1：physics 分支要做“拆开验证”

当前 physics 分支和 VMD-only 几乎重合，所以建议不要再把所有 physics 细节绑在一起试。

下一轮更应该拆成：

1. **只开 smoothness**
   - 例如：`lambda_smooth = 0.001 ~ 0.003`
   - `lambda_roll = 0`
2. **只开 roll consistency**
   - 例如：`lambda_roll = 0.005 ~ 0.01`
   - `lambda_smooth = 0`
3. **再开二者组合**

这样才能回答：

- 到底是哪一个物理项在帮忙；
- 还是二者都没有明显帮助。

### P1：继续盯住 `u / v`

下一轮不要只看整体 `rmse_mean`，必须同步盯：

- `rmse_u`
- `rmse_v`

因为从目前两轮结果看，**这些局部变量的恢复情况比总均值更能说明问题有没有真的被修正**。

### P2：为 physics 做统一口径复算

建议补一个离线分析步骤：

1. 用当前 `result_2_followup` 的 physics 指标定义；
2. 对 `result_1` 的 `predictions_*.npz` 重新计算：
   - `smoothness`
   - `roll_consistency_rmse`

这样后面就能真正判断：

- physics 微调到底是只改变了指标口径；
- 还是确实改善了物理一致性。

---

## 9.10 如果下一轮算力有限，最值得补的实验

如果你不想一下子铺很多实验，优先补下面 4 个最值钱：

1. `vmd_ccg_xlstm` with `lambda_vmd=0.05`
2. `vmd_ccg_xlstm` with `lambda_vmd=0.10`
3. `vmd_ccg_phys_xlstm` with only `lambda_roll`
4. `vmd_ccg_phys_xlstm` with only `lambda_smooth`

这 4 个实验能最快回答：

1. 当前问题是不是主要来自 **VMD 最终权重过大**；
2. physics 到底有没有哪一个子约束是有用的；
3. VMD 主线是否还有机会接近甚至追上 `CCG-xLSTM`。


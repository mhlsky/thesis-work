# 实验4手册：Phys / VMD 精修与联合验证

> 适用时间：`result3` 分析之后  
> 目标结果目录：`results/result4/`  
> 推荐环境：**Linux 云服务器**（先手动激活你自己的 Python / torch 环境）  
> 本轮目标：不是继续把所有模块直接堆成“最终模型”，而是先把 `result3` 中已经出现的有效方向系统收敛，再做少量联合验证。

---

## 1. 为什么要做实验4

经过 `result_1`、`result_2_followup`、`result3` 三轮实验，现在已经比较清楚：

1. `CCG-xLSTM` 仍然是稳定强基线；
2. 低权重 physics 约束**可能改善 OOD 泛化**；
3. VMD 对 **routine** 场景有明显收益，但 OOD 表现对：
   - 学习率；
   - `lambda_vmd`；
   - `state mixer`；
   - warmup；
   很敏感；
4. 当前“主误差最优”与“物理指标最优”还不是同一个配置；
5. 因此，实验4应该先把 **Phys** 和 **VMD** 各自收敛，再尝试小规模联合，而不是直接重新堆最终模型。

> 说明（2026-05-30 起脚本默认策略）：
>
> - `bash scripts/run_result4_experiments.sh . result4` 默认执行 **`RUN_GROUP=next_round`**；
> - 默认不会把 `robust` 多 seed 组一起带上；
> - VMD / Joint 类重模型默认采用 `batch_size=768`；
> - 若检测到 `results/result4_single_gpu/outputs/cache/vmd/K3_alpha2000/` 存在且完整，脚本会默认复用该 cache 并跳过重建。

---

## 2. 实验4要回答的核心问题

实验4主要回答 4 个问题：

1. **Phys 最合适的权重组合是什么？**
2. **当前 Phys 的 OOD 提升到底主要来自 roll、smooth，还是二者的小权重组合？**
3. **VMD 的主要收益到底来自更小学习率，还是去掉 / 弱化 state mixer？**
4. **当单模块 winner 确认后，Phys 与 VMD 联合时还能否继续提升？**

---

## 3. 实验4的设计原则

1. **不改数据划分**
   - 继续沿用当前 `train / validation / routine_test / ood_test`。

2. **先单模块收敛，再做联合**
   - 先分别做 Phys 精修和 VMD 精修；
   - 不再一开始就把 Phys 和 VMD 绑在一起解释。

3. **本轮重跑统一 baseline**
   - `result4` 必须有自己的 `CCG baseline`，避免跨轮直接对照带来口径混淆。

4. **训练监控口径要升级**
   - 当前 `best.pt` 的保存标准是 `val_rmse_mean`；
   - 因此分析时不能只看 `val_loss`，建议后续补充记录：
     - `val_pred_loss`
     - `val_vmd_aux_loss`
     - `val_phys_loss`
     - raw `smoothness`
     - raw `roll_consistency_rmse`

5. **正式结论必须做多随机种子**
   - 单次 seed 只用于宽扫和筛选；
   - 正式结论至少需要 5 seed。

---

## 4. 实验分组总览

从“实验内容”看，实验4分成 5 组：

1. 基线组
2. Phys 系统扫描组
3. VMD 系统扫描组
4. 联合验证组
5. 多随机种子稳健性组

说明：

- `configs/ablation_result4.yaml` 当前默认覆盖 **单 seed 主实验**；
- 多 seed `robust` 组会由脚本生成带 `seed` 后缀的运行时配置，但**不会自动进入默认主表汇总**；
- 这样做是为了避免把主表和 seed 扫描表混在一起。

从“脚本运行入口”看，当前更推荐把它拆成 4 层：

1. `next_round`：下一轮窄实验（默认）
2. `main_lite`：上一轮精简单 seed 主实验
3. `diag`：诊断组
4. `robust`：多 seed 稳健性复验

---

## 5. 基线组

| order | run_name | 目的 |
|---|---|---|
| 0 | `e4_ccg_xlstm_base` | 本轮统一对照基线 |

---

## 6. Phys 系统扫描组

### 6.1 固定设置

所有 Phys 实验默认统一使用：

- backbone：`CCG-xLSTM`
- `warmup_epochs = 8`
- `smoothness_target_cols = [p, r, phi]`
- `normalize_by_target_std = true`
- `roll_integration = trapezoid`
- `dt = 1.0`

### 6.2 主扫描组

| order | run_name | 主要配置 | 目的 |
|---|---|---|---|
| 10 | `e4_ccg_phys_s0005_r0000` | smooth=0.0005, roll=0 | 超轻 smooth |
| 11 | `e4_ccg_phys_s0010_r0000` | smooth=0.0010, roll=0 | 复验 smooth-only |
| 12 | `e4_ccg_phys_s0000_r0025` | smooth=0, roll=0.0025 | 超轻 roll |
| 13 | `e4_ccg_phys_s0000_r0050` | smooth=0, roll=0.0050 | 复验 roll-only |
| 14 | `e4_ccg_phys_s0000_r0075` | smooth=0, roll=0.0075 | 稍强 roll |
| 15 | `e4_ccg_phys_s0005_r0025` | 0.0005 / 0.0025 | 超保守组合 |
| 16 | `e4_ccg_phys_s0005_r0050` | 0.0005 / 0.0050 | 小 smooth + 中 roll |
| 17 | `e4_ccg_phys_s0010_r0050` | 0.0010 / 0.0050 | 复验当前 best OOD |
| 18 | `e4_ccg_phys_s0010_r0075` | 0.0010 / 0.0075 | 稍强组合 |
| 19 | `e4_ccg_phys_s0010_r0100` | 0.0010 / 0.0100 | 复验过强边界 |

### 6.3 Phys 诊断组

| order | run_name | 变化 | 目的 |
|---|---|---|---|
| 30 | `e4_ccg_phys_s0010_r0050_normoff` | `normalize_by_target_std=false` | 检查 loss / raw metric 脱钩 |
| 31 | `e4_ccg_phys_s0010_r0050_euler` | `roll_integration=euler` | 对比积分方式 |
| 32 | `e4_ccg_phys_s0010_r0050_warm4` | `warmup_epochs=4` | 更快介入 physics |
| 33 | `e4_ccg_phys_s0010_r0050_warm12` | `warmup_epochs=12` | 更慢介入 physics |

---

## 7. VMD 系统扫描组

### 7.1 固定设置

默认统一使用：

- backbone：`VMD-CCG-xLSTM`
- `K = 3`
- `alpha = 2000`
- `lambda_vmd = 0.05`
- `warmup_epochs = 5`
- physics 关闭

### 7.2 主扫描组

| order | run_name | 主要配置 | 目的 |
|---|---|---|---|
| 40 | `e4_vmd_ccg_l005_lr1e3_mix` | l=0.05, lr=1e-3, mixer=true | 复验低权重 VMD base |
| 41 | `e4_vmd_ccg_l005_lr5e4_mix` | l=0.05, lr=5e-4, mixer=true | 复验当前 OOD 最优 VMD |
| 42 | `e4_vmd_ccg_l005_lr1e3_nomix` | l=0.05, lr=1e-3, mixer=false | 复验当前 routine 最优结构 |
| 43 | `e4_vmd_ccg_l005_lr5e4_nomix` | l=0.05, lr=5e-4, mixer=false | 最关键组合实验 |
| 44 | `e4_vmd_ccg_l005_lr3e4_nomix` | l=0.05, lr=3e-4, mixer=false | 更稳学习率 |
| 45 | `e4_vmd_ccg_l005_lr7e4_nomix` | l=0.05, lr=7e-4, mixer=false | 略大学习率 |
| 46 | `e4_vmd_ccg_l002_lr5e4_nomix` | l=0.02, lr=5e-4, mixer=false | 更轻 VMD loss |
| 47 | `e4_vmd_ccg_l003_lr5e4_nomix` | l=0.03, lr=5e-4, mixer=false | 中低权重 |
| 48 | `e4_vmd_ccg_l008_lr5e4_nomix` | l=0.08, lr=5e-4, mixer=false | 检查较强边界 |
| 49 | `e4_vmd_ccg_l005_lr5e4_nomix_warm3` | warmup=3 | 更快拉起辅助监督 |
| 50 | `e4_vmd_ccg_l005_lr5e4_nomix_warm8` | warmup=8 | 更慢拉起辅助监督 |

### 7.3 弱 mixer 诊断组

| order | run_name | 变化 | 目的 |
|---|---|---|---|
| 51 | `e4_vmd_ccg_l005_lr5e4_h8_d01` | mixer hidden=8, dropout=0.1 | 弱 mixer 方案 A |
| 52 | `e4_vmd_ccg_l005_lr5e4_h4_d01` | mixer hidden=4, dropout=0.1 | 弱 mixer 方案 B |

### 7.4 alpha 微调组

| order | run_name | 变化 | 目的 |
|---|---|---|---|
| 53 | `e4_vmd_ccg_l005_lr5e4_nomix_a1500` | alpha=1500 | 更弱带宽约束 |
| 54 | `e4_vmd_ccg_l005_lr5e4_nomix_a2500` | alpha=2500 | 略强带宽约束 |

---

## 8. 联合验证组

联合组默认优先使用实验4当前假设的 VMD 候选主干：

- `lambda_vmd=0.05`
- `lr=5e-4`
- `use_state_mixer=false`

如果后续单模块筛选结果表明别的 VMD 配置更稳，再整体替换成新的 winner。

| order | run_name | 主要配置 | 目的 |
|---|---|---|---|
| 70 | `e4_joint_l005_lr5e4_nomix_r0025` | + roll=0.0025 | 最轻联合 |
| 71 | `e4_joint_l005_lr5e4_nomix_r0050` | + roll=0.0050 | roll-only 联合 |
| 72 | `e4_joint_l005_lr5e4_nomix_s0005_r0025` | + smooth 0.0005 / roll 0.0025 | 超保守联合 |
| 73 | `e4_joint_l005_lr5e4_nomix_s0005_r0050` | + smooth 0.0005 / roll 0.0050 | 温和联合 |
| 74 | `e4_joint_l005_lr5e4_nomix_s0010_r0050` | + smooth 0.001 / roll 0.005 | 迁移 Phys best |
| 75 | `e4_joint_l005_lr5e4_nomix_s0010_r0075` | + smooth 0.001 / roll 0.0075 | 检查联合时是否更易过约束 |

---

## 9. 多随机种子稳健性组

### 9.1 论文目标与脚本默认值

- 论文正式结论仍建议尽量做到 **5 seeds**；
- 但为了控制实验时长，脚本当前默认的 `robust` 只先跑 **3 seeds**：

- `7`
- `42`
- `2026`

### 9.2 默认候选

脚本默认会为下面这些候选生成多 seed 运行时配置：

1. `e4_ccg_xlstm_base`
2. `e4_ccg_phys_s0010_r0050`
3. `e4_vmd_ccg_l005_lr5e4_nomix`
4. `e4_joint_l005_lr5e4_nomix_r0050`

如果后续你确认了新的 winner，可以通过环境变量覆盖：

```bash
ROBUST_BASES=e4_ccg_xlstm_base,e4_ccg_phys_s0000_r0050,e4_vmd_ccg_l003_lr5e4_nomix \
bash scripts/run_result4_experiments.sh . result4
```

说明：

- 当前 `robust` 组会为每个 seed 生成**独立 run 目录**；
- 默认主表 `configs/ablation_result4.yaml` 不会把这些 seed run 自动纳入；
- 这样能先把单 seed 主表和 seed 扫描分离，避免混表。

---

## 10. 推荐执行顺序

### 第一阶段：下一轮窄实验

建议顺序：

1. `next_round`
2. 查看 VMD / Joint / Phys 少量验证结果
3. 如有必要，再补跑 `diag`

### 第二阶段：筛选 winner

建议保留：

- Phys：保留 2 个 physics 候选 + 1 个 `smooth(p,phi)` 诊断候选
- VMD：保留 2 个 OOD 候选 + 1 个 routine / 结构候选
- Joint：只验证 `lr=3e-4, no mixer` 主干上的 3 个候选

### 第三阶段：多 seed

对最终 shortlist 再跑 `robust` 组。

---

## 11. 运行方式

### 11.1 默认推荐入口（下一轮窄实验）

```bash
bash scripts/run_result4_experiments.sh . result4
```

说明：

- 这是当前默认推荐入口；
- 实际等价于 `RUN_GROUP=next_round`；
- 默认不会一起跑 `robust`；
- 如果检测到已有 `results/result4_single_gpu/outputs/cache/vmd/K3_alpha2000/`，会自动复用并跳过 VMD cache 重建。

### 11.2 下一轮窄实验（显式写法）

```bash
RUN_GROUP=next_round bash scripts/run_result4_experiments.sh . result4
```

### 11.3 上一轮精简主实验（回放）

```bash
RUN_GROUP=main_lite bash scripts/run_result4_experiments.sh . result4
```

### 11.4 跑诊断组

```bash
RUN_GROUP=diag bash scripts/run_result4_experiments.sh . result4
```

### 11.5 单 seed 全量主实验

```bash
RUN_GROUP=all bash scripts/run_result4_experiments.sh . result4
```

### 11.6 只跑 Phys 相关

```bash
RUN_GROUP=phys bash scripts/run_result4_experiments.sh . result4
```

### 11.7 只跑 VMD 相关

```bash
RUN_GROUP=vmd bash scripts/run_result4_experiments.sh . result4
```

### 11.8 只跑联合组

```bash
RUN_GROUP=joint bash scripts/run_result4_experiments.sh . result4
```

### 11.9 只跑多 seed 组

```bash
RUN_GROUP=robust bash scripts/run_result4_experiments.sh . result4
```

### 11.10 完整全量实验（单 seed 全量 + 当前默认 robust）

```bash
RUN_GROUP=full bash scripts/run_result4_experiments.sh . result4
```

说明：

- `full` 仍然存在，但不再是默认入口；
- 当前 `full = all + 精简后的 robust`；
- 如果你想恢复更大的多 seed 集合，需要额外覆盖 `ROBUST_BASES` / `ROBUST_SEEDS`。

### 11.11 只跑少数几个配置

```bash
RUN_ONLY=e4_ccg_xlstm_base,e4_vmd_ccg_l005_lr5e4_nomix,e4_joint_l005_lr5e4_nomix_r0050 \
bash scripts/run_result4_experiments.sh . result4
```

### 11.12 自定义 seed

```bash
RUN_GROUP=robust ROBUST_SEEDS=11,42,1234 bash scripts/run_result4_experiments.sh . result4
```

---

## 12. 输出目录说明

```text
results/result4/
  logs/
  outputs/
    e4_ccg_xlstm_base/
    e4_ccg_phys_.../
    e4_vmd_ccg_.../
    e4_joint_.../
    summary/
      ablation_routine_test.csv
      ablation_ood_test.csv
      physics_metrics.csv
```

说明：

- `summary/` 默认只汇总 `configs/ablation_result4.yaml` 中的单 seed 主实验；
- `robust` 组的 seed run 不会自动进入这张主表；
- 如果后续你要做 seed 聚合，可以再单独扩展结果汇总脚本。

---

## 13. 分析时优先看什么

### 主指标

1. `OOD rmse_mean`
2. `OOD rmse_u`
3. `OOD rmse_v`

### 次指标

4. `routine rmse_mean`
5. `routine rmse_u`
6. `routine rmse_v`

### 物理指标

7. `smoothness`
8. `roll_consistency_rmse`

### 训练稳定性

9. `best_epoch`（以 `val_rmse_mean` 为准）
10. `train_log.csv` 中 `val_rmse_mean` 是否稳定

---

## 14. 实验4的成功标准

### Phys 成功

满足任一条即可：

1. 多 seed 平均 OOD RMSE 稳定优于 baseline；
2. OOD RMSE 基本不变，但 physics 指标稳定更优；
3. `u/v` 未明显恶化，同时 `roll_consistency_rmse` 更稳。

### VMD 成功

满足任一条即可：

1. 多 seed 平均 OOD RMSE 不差于 baseline；
2. routine RMSE 显著优于 baseline，且 OOD 不退化；
3. 相比 `result_2_followup`，训练稳定性明显改善。

### Joint 成功

满足任一条即可：

1. OOD 优于单模块 best；
2. routine / OOD 折中优于单模块；
3. physics 指标与 OOD 指标不再明显冲突。

---

## 15. 最后建议

实验4最关键的不是“把实验继续做复杂”，而是把 `result3` 已经出现的正信号做实：

- Phys：确认 **小权重 roll / small smooth+roll** 哪条更稳；
- VMD：确认 **小学习率 + 去 mixer / 弱 mixer** 是否是主因；
- Joint：只在单模块 winner 上做少量联合；
- Formal 结论：必须用多 seed 支撑。

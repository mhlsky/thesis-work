# Step 01~07 实验运行手册

> 这份手册面向第一次接触本项目的同学，目标是把 **Step 01 到 Step 07** 的可运行命令、推荐顺序、输出位置和常见问题讲清楚。  
> 默认情况下：**本地 Windows 开发继续使用 PowerShell + `uv`**；**Linux 云服务器建议先激活你自己的 Python/torch 环境，再直接用 `python` 运行**。如果你只想先验证流程是否跑通，请优先使用 **smoke** 命令。

---

## 1. 先理解本项目的运行原则

本项目把实现过程拆成 Step 01 ~ Step 08。  
其中：

- **Step 01~06**：逐步把数据、模型、损失和最终主模型做出来；
- **Step 07**：不再新增模型，转为做 **消融实验、汇总结果、生成论文图表**；
- **Step 08**：线性注意力，可选增强，不属于主线最低必做。

### 1.1 两类运行方式

你会看到两类命令：

1. **Smoke 命令**
   - 作用：只验证流程
   - 特点：样本少、batch 小、step 少、epoch 少
   - 适合：本地电脑、第一次检查代码是否能跑通

2. **正式命令**
   - 作用：跑正式训练、正式评估、正式汇总
   - 特点：会更慢，输出更完整
   - 适合：主线功能稳定后，在本地高配机器或租用算力平台运行

### 1.2 推荐顺序

如果你是第一次完整跑项目，建议顺序是：

```text
Step 00 环境准备
Step 01 数据流程 smoke
Step 02 基线训练 smoke
Step 03 VMD smoke
Step 04 Lite-xLSTM / CCG-xLSTM smoke
Step 05 VMD-CCG-xLSTM smoke
Step 06 最终主模型 smoke
Step 07 结果汇总与绘图 smoke
```

确认 smoke 都通过后，再选择需要的正式命令。

---

## 2. 环境准备（所有步骤之前先做）

### 2.1 安装依赖

在仓库根目录执行：

```powershell
uv sync
```

作用：

- 创建 / 同步 `.venv`
- 安装 `torch`、`numpy`、`scikit-learn`、`vmdpy`
- 安装 Step 07 绘图依赖 `matplotlib`

### 2.2 推荐检查

检查 Python 是否来自 `uv` 环境：

```powershell
uv run python --version
```

检查项目是否能正常导入：

```powershell
uv run python -c "import ship_motion; print('import ok')"
```

### 2.3 关于 `pwsh` 和 `powershell`

如果你安装了 PowerShell 7，通常可以使用：

```powershell
pwsh -File .\scripts\xxx.ps1
```

如果你当前机器上只有 Windows PowerShell，也可以使用：

```powershell
powershell -File .\scripts\xxx.ps1
```

本项目脚本内容本身不依赖一定要 `pwsh`，只是命令入口名称可能不同。

### 2.4 云上 Linux 入口脚本

如果你准备把正式训练放到 Linux 云服务器上，现在仓库里已经提供：

```text
scripts/setup_cloud_env.sh
scripts/run_full_training.sh
```

如果你希望训练在 SSH 断开后继续运行，建议再配合阅读：

```text
docs/tmux_cloud_manual.md
```

推荐顺序：

```bash
conda activate your_env
bash scripts/setup_cloud_env.sh
bash scripts/run_full_training.sh
```

说明：

- 请先手动激活你自己的 conda / venv 环境；
- 该环境需要提前装好 `torch`、`numpy`、`scikit-learn`、`vmdpy`、`matplotlib` 等依赖；
- 这两个 Linux 脚本现在会**直接复用当前激活环境里的 `python`**，不再默认执行 `uv sync` 或 `uv run`。

第一个脚本负责环境检查与 smoke 检查；  
第二个脚本负责正式 VMD 缓存、正式训练、predictions 导出、汇总与绘图。

---

## 3. Step 01：数据读取、滑窗、标准化

### 3.1 这一步做什么

Step 01 负责：

- 读取 CSV
- 按 `seq_len / pred_len` 切滑动窗口
- 拟合训练集标准化器
- 返回：
  - `x`
  - `x_exog`
  - `x_state`
  - `y`
  - `y_raw`
  - `last_state_raw`

### 3.2 推荐先跑 smoke

命令：

```powershell
uv run python -m ship_motion.data.dataset --config configs/base.yaml --smoke
```

或脚本：

```powershell
powershell -File .\scripts\smoke_step_01_data.ps1
```

### 3.3 正式检查命令

如果你想看完整数据集能否正常构建：

```powershell
uv run python -m ship_motion.data.dataset --config configs/base.yaml
```

### 3.4 通过标准

你应该能看到类似：

- train / val / test 文件数
- train / val 窗口数
- `x shape`
- `x_exog shape`
- `x_state shape`
- `y shape`

---

## 4. Step 02：基线模型训练与评估

### 4.1 这一步做什么

Step 02 实现并统一管理基础模型：

- Persistence
- LSTM
- GRU
- TCN（可选）
- Transformer

并提供统一：

- 训练入口
- 验证 / 测试评估
- checkpoint 保存
- 指标保存

### 4.2 单模型 smoke

例如先跑 LSTM：

```powershell
uv run python -m ship_motion.train --config configs/lstm.yaml --smoke
```

### 4.3 批量 baseline smoke

```powershell
powershell -File .\scripts\smoke_step_02_training.ps1
```

### 4.4 正式训练示例

```powershell
uv run python -m ship_motion.train --config configs/lstm.yaml
uv run python -m ship_motion.train --config configs/gru.yaml
uv run python -m ship_motion.train --config configs/transformer.yaml
```

### 4.5 正式评估示例

```powershell
uv run python -m ship_motion.evaluate --config configs/lstm.yaml --checkpoint outputs/lstm_seq128_pred10/best.pt --split all
```

### 4.6 常见输出目录

例如 LSTM：

```text
outputs/lstm_seq128_pred10/
  best.pt
  config.yaml
  scaler.json
  train_log.csv
  metrics_val.json
  metrics_routine_test.json
  metrics_ood_test.json
```

smoke 时会变成：

```text
outputs/smoke/lstm_seq128_pred10_smoke/
```

---

## 5. Step 03：VMD 模态缓存构建

### 5.1 这一步做什么

Step 03 不训练模型，主要做：

- 对目标状态序列做 VMD 分解
- 生成缓存 `.npz`
- 让 Dataset 返回 `y_modes`

注意：

- `y_modes` **只作为辅助标签**
- **不作为模型输入**

### 5.2 推荐先跑 smoke

```powershell
uv run python -m ship_motion.data.vmd --config configs/vmd_ccg_xlstm.yaml --smoke
```

或脚本：

```powershell
powershell -File .\scripts\smoke_step_03_vmd.ps1
```

### 5.3 正式构建缓存

```powershell
powershell -File .\scripts\build_vmd_cache.ps1
```

如果你想手动指定缓存目录：

```powershell
powershell -File .\scripts\build_vmd_cache.ps1 -CacheRoot outputs/cache/vmd/K3_alpha2000
```

### 5.4 典型输出

```text
outputs/cache/vmd/K3_alpha2000/
  train/
  validation/
  routine_test/
  ood_test/
```

smoke 默认写到：

```text
outputs/cache/vmd_smoke/K3_alpha2000/
```

---

## 6. Step 04：Lite-xLSTM 与 CCG-xLSTM

### 6.1 这一步做什么

Step 04 引入：

- `Lite-xLSTM`
- `CCG-xLSTM`

核心点：

- 指数门控
- 归一化记忆更新
- `CCG`：把控制量 / 风场量直接送进门控

### 6.2 smoke 命令

```powershell
powershell -File .\scripts\smoke_step_04_xlstm.ps1
```

### 6.3 单模型训练

Lite-xLSTM：

```powershell
powershell -File .\scripts\train_lite_xlstm.ps1
```

CCG-xLSTM：

```powershell
powershell -File .\scripts\train_ccg_xlstm.ps1
```

如果只是 smoke：

```powershell
powershell -File .\scripts\train_lite_xlstm.ps1 -Smoke
powershell -File .\scripts\train_ccg_xlstm.ps1 -Smoke
```

### 6.4 直接命令版

```powershell
uv run python -m ship_motion.train --config configs/lite_xlstm.yaml
uv run python -m ship_motion.train --config configs/ccg_xlstm.yaml
```

---

## 7. Step 05：VMD-CCG-xLSTM 集成

### 7.1 这一步做什么

Step 05 在 Step 04 基础上加入：

- `VMDMultiHead`
- `DeltaDecoder`
- `StateCouplingMixer`

形成：

```text
VMD-CCG-xLSTM
```

### 7.2 推荐 smoke

```powershell
powershell -File .\scripts\smoke_step_05_vmd_ccg.ps1
```

这个脚本会自动：

1. 先构建最小 VMD smoke 缓存
2. 再跑 `VMD-CCG-xLSTM` 的 smoke 训练

### 7.3 正式训练

```powershell
powershell -File .\scripts\train_vmd_ccg_xlstm.ps1
```

只做 smoke：

```powershell
powershell -File .\scripts\train_vmd_ccg_xlstm.ps1 -Smoke
```

### 7.4 直接命令版

```powershell
uv run python -m ship_motion.train --config configs/vmd_ccg_xlstm.yaml
```

---

## 8. Step 06：物理约束与最终主模型

### 8.1 这一步做什么

Step 06 在 Step 05 的基础上加入：

- 平滑性约束
- 横摇运动学约束

最终形成训练层面的主模型：

```text
VMD-CCG-Phys-xLSTM
```

注意：

- 网络结构主体仍然是 `vmd_ccg_xlstm`
- “Phys” 主要体现在：
  - `physics loss`
  - `physics metrics`
  - `final config`

### 8.2 推荐 smoke

```powershell
powershell -File .\scripts\smoke_step_06_physics.ps1
```

### 8.3 正式训练最终模型

```powershell
powershell -File .\scripts\train_final_model.ps1
```

只跑 smoke：

```powershell
powershell -File .\scripts\train_final_model.ps1 -Smoke
```

### 8.4 直接命令版

```powershell
uv run python -m ship_motion.train --config configs/vmd_ccg_phys_xlstm.yaml
```

### 8.5 这一步新增的物理指标

评估文件里会出现：

- `smoothness`
- `roll_consistency_rmse`

它们用于说明：

- 预测是否更平滑
- `p` 与 `phi` 是否更符合物理关系

---

## 9. Step 07：主线消融实验、结果汇总与绘图

### 9.1 这一步做什么

Step 07 不再新增模型，而是做：

- 主线模型消融对比
- 结果汇总
- 物理指标表
- 预测曲线图
- 柱状图

### 9.2 推荐 smoke

```powershell
powershell -File .\scripts\smoke_step_07_experiments.ps1
```

这个脚本会：

1. 复用已有 smoke run
2. 重新导出 `predictions_*.npz`
3. 汇总 CSV
4. 生成图表

### 9.3 批量导出 predictions

```powershell
powershell -File .\scripts\evaluate_all.ps1 -Smoke
```

正式版本：

```powershell
powershell -File .\scripts\evaluate_all.ps1
```

### 9.4 汇总与绘图

Smoke：

```powershell
powershell -File .\scripts\plot_results.ps1 -Smoke
```

正式：

```powershell
powershell -File .\scripts\plot_results.ps1
```

### 9.5 批量主线训练

如果你要顺序跑主线实验：

```powershell
powershell -File .\scripts\run_ablation.ps1
```

如果只是 smoke：

```powershell
powershell -File .\scripts\run_ablation.ps1 -Smoke
```

### 9.6 Step 07 典型输出

```text
outputs/summary/
  ablation_routine_test.csv
  ablation_ood_test.csv
  physics_metrics.csv
  figures/
    pred_u_routine.png
    pred_phi_routine.png
    pred_u_ood.png
    pred_phi_ood.png
    rmse_bar.png
    roll_consistency_bar.png
```

smoke 默认写到：

```text
outputs/smoke/summary/
```

---

## 10. 一组最常用命令速查

如果你不想一开始看太多，只想先照着跑，推荐按下面顺序：

### 10.1 初始化环境

```powershell
uv sync
```

### 10.2 数据 smoke

```powershell
powershell -File .\scripts\smoke_step_01_data.ps1
```

### 10.3 基线 smoke

```powershell
powershell -File .\scripts\smoke_step_02_training.ps1
```

### 10.4 VMD smoke

```powershell
powershell -File .\scripts\smoke_step_03_vmd.ps1
```

### 10.5 xLSTM smoke

```powershell
powershell -File .\scripts\smoke_step_04_xlstm.ps1
```

### 10.6 VMD-CCG-xLSTM smoke

```powershell
powershell -File .\scripts\smoke_step_05_vmd_ccg.ps1
```

### 10.7 最终主模型 smoke

```powershell
powershell -File .\scripts\smoke_step_06_physics.ps1
```

### 10.8 汇总与绘图 smoke

```powershell
powershell -File .\scripts\smoke_step_07_experiments.ps1
```

---

## 11. 输出目录怎么理解

### 11.1 `outputs/smoke/`

这里放本地轻量验证结果。  
特点：

- 可反复覆盖
- 不代表正式结论
- 用来验证流程和代码

### 11.2 `outputs/{run_name}/`

这里放正式训练结果。  
常见文件：

- `best.pt`
- `config.yaml`
- `scaler.json`
- `train_log.csv`
- `metrics_val.json`
- `metrics_routine_test.json`
- `metrics_ood_test.json`
- `predictions_routine_test.npz`
- `predictions_ood_test.npz`

### 11.3 `outputs/cache/`

这里放 VMD 缓存。  
正式和 smoke 分开：

- 正式：`outputs/cache/vmd/...`
- smoke：`outputs/cache/vmd_smoke/...`

---

## 12. 常见问题与排查建议

### 12.1 `pwsh` 找不到

改用：

```powershell
powershell -File .\scripts\xxx.ps1
```

### 12.2 `vmdpy` / `torch` / `matplotlib` 导入失败

先执行：

```powershell
uv sync
```

### 12.3 VMD 模型报缓存找不到

先跑：

```powershell
powershell -File .\scripts\smoke_step_03_vmd.ps1
```

或正式：

```powershell
powershell -File .\scripts\build_vmd_cache.ps1
```

### 12.4 训练很慢

优先先跑 smoke。  
如果正式训练仍慢：

- 减少 `batch_size`
- 减少 `num_layers`
- 减少 `d_model`
- 在本地只做 smoke，把正式实验迁移到算力平台

### 12.5 xLSTM 训练出现 NaN

优先尝试：

- `lr: 1e-3 -> 5e-4`
- `gate_clip: 5.0 -> 3.0`
- `context_dim: 64 -> 32`
- `num_layers: 2 -> 1`

### 12.6 物理约束导致误差变差

这是可能出现的。建议按顺序：

1. 先确认 Step 05 正常
2. `lambda_smooth = 0.001`
3. 再加 `lambda_roll = 0.01`
4. 最后再提高到正式权重

不要一开始把物理 loss 权重设太大。

---

## 13. 对论文写作最有用的几个目录

### 13.1 看单个模型结果

```text
outputs/{run_name}/
```

### 13.2 看主线汇总表

```text
outputs/summary/
```

### 13.3 看论文图

```text
outputs/summary/figures/
```

---

## 14. 最后建议

如果你是第一次完整接手这个项目，不要一开始就跑正式全量实验。  
最稳妥的方法是：

1. **先全部 smoke 跑通**
2. **再选关键模型做正式训练**
3. **最后用 Step 07 汇总和画图**

这样最省时间，也最不容易卡在半路。

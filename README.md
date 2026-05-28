# Thesis Work

船舶运动姿态多步预测实验项目，使用 `uv` 管理环境，采用标准 `src/` 布局，面向 PyCharm 与命令行协同开发。

## 项目结构

```text
thesis-work/
  README.md
  pyproject.toml
  uv.lock
  configs/
    base.yaml
    persistence.yaml
    lstm.yaml
    gru.yaml
    tcn.yaml
    transformer.yaml
    vmd_ccg_xlstm.yaml
  data/
  docs/
    project_overview.md
    learn.md
    implementation_steps/
    references/
  results/
  outputs/
  scripts/
    smoke_step_01_data.ps1
    smoke_step_02_training.ps1
    smoke_step_03_vmd.ps1
    build_vmd_cache.ps1
    run_baselines.ps1
    run_formal_result.ps1
    run_followup_experiments.ps1
    run_followup_experiments.sh
  src/
    ship_motion/
      data/
        dataset.py
        scaler.py
        vmd.py
      losses/
        vmd_loss.py
      models/
        persistence.py
        lstm.py
        gru.py
        tcn.py
        transformer.py
      evaluate.py
      metrics.py
      train.py
      utils.py
```

## 常用命令

```powershell
uv sync
uv run python -m ship_motion.data.dataset --config configs/base.yaml --smoke
uv run python -m ship_motion.train --config configs/lstm.yaml --smoke
uv run python -m ship_motion.data.vmd --config configs/vmd_ccg_xlstm.yaml --smoke
pwsh -File .\scripts\smoke_step_01_data.ps1
pwsh -File .\scripts\smoke_step_02_training.ps1
pwsh -File .\scripts\smoke_step_03_vmd.ps1
pwsh -File .\scripts\build_vmd_cache.ps1
pwsh -File .\scripts\run_baselines.ps1
pwsh -File .\scripts\run_followup_experiments.ps1 -ResultName result_2_followup
bash scripts/run_followup_experiments.sh . result_2_followup
bash scripts/run_result3_experiments.sh . result3
bash scripts/run_result4_experiments.sh . result4
```

## Step 01~07 实验手册

如果你想按步骤完整跑通本项目，建议直接看：

- `docs/experiment_manual.md`

这份手册已经整理了：

- Step 01~07 的推荐运行顺序
- 每一步的 smoke / 正式命令
- 输出目录说明
- Step 07 的汇总与绘图命令
- 常见报错与排查建议

## 本地验证原则

- 每个实现步骤都必须配套一个轻量 smoke-test 脚本或等价命令。
- 本地 smoke test 只验证流程是否跑通，不要求完整训练效果。
- smoke test 应使用极小样本、极少 epoch/step、小 batch，并把临时输出写入 `outputs/`。
- 正式训练、消融和完整评估默认写入 `results/<result_name>/`，便于归档和重画图表。

## 训练脚本的硬件感知默认值

- 训练入口现在支持 `batch_size / num_workers / prefetch_factor / precision = auto`。
- 正式 Linux / Windows 训练脚本会自动加载：
  - `scripts/server_train_env.sh`
  - `scripts/server_train_env.ps1`
- 当前默认面向 **1 卡 40GB GPU + 12 核 CPU + 32GB 内存** 的服务器档位：
  - 模型相关 `batch_size` 会按模型族自动放大；
  - `num_workers` 会按 CPU / 内存自动推断；
  - CUDA 且支持时会优先使用 `bf16`，否则回退到 `fp32`。
- 如果后续换机器，可在运行前临时覆盖：

```bash
export SHIP_MOTION_HW_PROFILE=single_gpu_24g
export SHIP_MOTION_GPU_MEMORY_GB=24
export SHIP_MOTION_CPU_CORES=8
export SHIP_MOTION_SYSTEM_MEMORY_GB=24
export SHIP_MOTION_PRECISION=fp32
```

## 文档入口

- 总体方案：`docs/project_overview.md`
- 入门说明：`docs/learn.md`
- 实验手册：`docs/experiment_manual.md`
- 实验3手册：`docs/experiment3_manual.md`
- 实验4手册：`docs/experiment4_manual.md`
- 云上 tmux 手册：`docs/tmux_cloud_manual.md`
- 分步实现：`docs/implementation_steps/README.md`
- Step 02 说明：`docs/implementation_steps/02_training_lstm.md`
- Step 03 说明：`docs/implementation_steps/03_vmd_module.md`

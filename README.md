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
  data/
  docs/
    project_overview.md
    learn.md
    implementation_steps/
    references/
  outputs/
  scripts/
    smoke_step_01_data.ps1
    smoke_step_02_training.ps1
    run_baselines.ps1
  src/
    ship_motion/
      data/
        dataset.py
        scaler.py
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
pwsh -File .\scripts\smoke_step_01_data.ps1
pwsh -File .\scripts\smoke_step_02_training.ps1
pwsh -File .\scripts\run_baselines.ps1
```

## 本地验证原则

- 每个实现步骤都必须配套一个轻量 smoke-test 脚本或等价命令。
- 本地 smoke test 只验证流程是否跑通，不要求完整训练效果。
- smoke test 应使用极小样本、极少 epoch/step、小 batch，并把临时输出写入 `outputs/`。
- 正式训练、消融和完整评估默认后续放到租用算力平台运行。

## 文档入口

- 总体方案：`docs/project_overview.md`
- 入门说明：`docs/learn.md`
- 分步实现：`docs/implementation_steps/README.md`
- Step 02 说明：`docs/implementation_steps/02_training_lstm.md`

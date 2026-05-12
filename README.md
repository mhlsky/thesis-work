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
  data/
  docs/
    project_overview.md
    learn.md
    implementation_steps/
    references/
  outputs/
  scripts/
    smoke_test.ps1
  src/
    ship_motion/
      data/
        dataset.py
        scaler.py
      utils.py
```

## 常用命令

```powershell
uv sync
uv run python -m ship_motion.data.dataset --config configs/base.yaml --smoke
pwsh -File .\scripts\smoke_test.ps1
```

## 文档入口

- 总体方案：`docs/project_overview.md`
- 入门说明：`docs/learn.md`
- 分步实现：`docs/implementation_steps/README.md`

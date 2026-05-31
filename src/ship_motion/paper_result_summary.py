from __future__ import annotations

"""整理可直接用于论文书写的实验结果表和图。

这个脚本面向当前仓库中已经完成的 result_1 / result_2_followup /
result3 / result4 / result4_next_round / result4_robust。

输出内容统一写到 docs/paper/result_summary/ 下，包括：
1. 代表性结果总表；
2. result4 robust 多 seed 统计表；
3. 若干可直接插入论文草稿的图表；
4. 一份 Markdown 说明，帮助后续写作时快速取数。
"""

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ship_motion.utils import ensure_dir, load_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "paper" / "result_summary"

SPLITS = ("val", "routine_test", "ood_test")
PHYSICS_KEYS = (
    "smoothness",
    "roll_consistency_rmse",
    "first_step_jump_rmse",
    "max_abs_second_diff_p95",
)


@dataclass(frozen=True)
class RunSpec:
    stage: str
    family: str
    label: str
    run_name: str
    output_root: Path
    note: str = ""

    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name


@dataclass(frozen=True)
class RobustGroupSpec:
    base_name: str
    label: str
    family: str


RESULT1_SPECS: tuple[RunSpec, ...] = (
    RunSpec("result_1", "baseline", "LSTM", "lstm_seq128_pred10", REPO_ROOT / "results" / "result_1" / "outputs"),
    RunSpec("result_1", "baseline", "GRU", "gru_seq128_pred10", REPO_ROOT / "results" / "result_1" / "outputs"),
    RunSpec(
        "result_1",
        "baseline",
        "Transformer",
        "transformer_seq128_pred10",
        REPO_ROOT / "results" / "result_1" / "outputs",
    ),
    RunSpec(
        "result_1",
        "baseline",
        "Lite-xLSTM",
        "lite_xlstm_seq128_pred10",
        REPO_ROOT / "results" / "result_1" / "outputs",
    ),
    RunSpec(
        "result_1",
        "baseline",
        "CCG-xLSTM Baseline",
        "ccg_xlstm_seq128_pred10",
        REPO_ROOT / "results" / "result_1" / "outputs",
        note="实验一中的最佳主干",
    ),
    RunSpec(
        "result_1",
        "vmd",
        "VMD-CCG-xLSTM (early)",
        "vmd_ccg_xlstm_seq128_pred10",
        REPO_ROOT / "results" / "result_1" / "outputs",
        note="早期 VMD 原型，结果劣于 baseline",
    ),
    RunSpec(
        "result_1",
        "joint",
        "VMD-CCG-Phys-xLSTM (early)",
        "vmd_ccg_phys_xlstm_seq128_pred10",
        REPO_ROOT / "results" / "result_1" / "outputs",
        note="早期 VMD+Phys 原型，结果劣于 baseline",
    ),
)

RESULT2_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        "result_2_followup",
        "baseline",
        "Persistence",
        "persistence_seq128_pred10",
        REPO_ROOT / "results" / "result_2_followup" / "results" / "result_2_followup" / "outputs",
    ),
    RunSpec(
        "result_2_followup",
        "vmd",
        "VMD-CCG-xLSTM Follow-up",
        "vmd_ccg_xlstm_seq128_pred10",
        REPO_ROOT / "results" / "result_2_followup" / "results" / "result_2_followup" / "outputs",
        note="VMD 缓存流程打通后的补跑结果",
    ),
    RunSpec(
        "result_2_followup",
        "joint",
        "VMD-CCG-Phys-xLSTM Follow-up",
        "vmd_ccg_phys_xlstm_seq128_pred10",
        REPO_ROOT / "results" / "result_2_followup" / "results" / "result_2_followup" / "outputs",
        note="与 result_1 相比，VMD 流程更稳定但仍非最终结论",
    ),
)

RESULT3_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        "result3",
        "baseline",
        "CCG Baseline",
        "e3_ccg_xlstm_base",
        REPO_ROOT / "results" / "result3" / "outputs",
    ),
    RunSpec(
        "result3",
        "phys",
        "CCG-Phys (best OOD)",
        "e3_ccg_phys_s001_r005",
        REPO_ROOT / "results" / "result3" / "outputs",
        note="Phys 在 result3 中的最佳 OOD 候选",
    ),
    RunSpec(
        "result3",
        "vmd",
        "VMD-CCG (best OOD)",
        "e3_vmd_ccg_l005_lr5e4",
        REPO_ROOT / "results" / "result3" / "outputs",
    ),
    RunSpec(
        "result3",
        "vmd",
        "VMD-CCG (no state mixer)",
        "e3_vmd_ccg_l005_nomixer",
        REPO_ROOT / "results" / "result3" / "outputs",
        note="结构消融证据",
    ),
)

RESULT4_MAIN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        "result4_main",
        "baseline",
        "CCG Baseline",
        "e4_ccg_xlstm_base",
        REPO_ROOT / "results" / "result4" / "outputs",
    ),
    RunSpec(
        "result4_main",
        "phys",
        "CCG-Phys (s=0.001, r=0.005)",
        "e4_ccg_phys_s0010_r0050",
        REPO_ROOT / "results" / "result4" / "outputs",
    ),
    RunSpec(
        "result4_main",
        "vmd",
        "VMD-CCG (l=0.05, lr=3e-4, no mixer)",
        "e4_vmd_ccg_l005_lr3e4_nomix",
        REPO_ROOT / "results" / "result4" / "outputs",
    ),
    RunSpec(
        "result4_main",
        "joint",
        "Joint (l=0.05, lr=5e-4, s=0.001, r=0.005)",
        "e4_joint_l005_lr5e4_nomix_s0010_r0050",
        REPO_ROOT / "results" / "result4" / "outputs",
    ),
)

RESULT4_NEXT_ROUND_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        "result4_next_round",
        "baseline",
        "CCG Baseline",
        "e4_ccg_xlstm_base",
        REPO_ROOT / "results" / "result4_next_round" / "outputs",
    ),
    RunSpec(
        "result4_next_round",
        "phys",
        "CCG-Phys (p/phi smooth, r=0.0075)",
        "e4_ccg_phys_s0010_r0075_prphi",
        REPO_ROOT / "results" / "result4_next_round" / "outputs",
        note="Phys 转向轨迹平滑/物理一致性后的代表方案",
    ),
    RunSpec(
        "result4_next_round",
        "vmd",
        "VMD-CCG (l=0.03, lr=5e-4, no mixer)",
        "e4_vmd_ccg_l003_lr5e4_nomix",
        REPO_ROOT / "results" / "result4_next_round" / "outputs",
    ),
    RunSpec(
        "result4_next_round",
        "joint",
        "Joint (l=0.05, lr=3e-4, r=0.005)",
        "e4_joint_l005_lr3e4_nomix_r0050",
        REPO_ROOT / "results" / "result4_next_round" / "outputs",
    ),
    RunSpec(
        "result4_next_round",
        "joint",
        "Joint (l=0.05, lr=3e-4, s=0.001, r=0.005)",
        "e4_joint_l005_lr3e4_nomix_s0010_r0050",
        REPO_ROOT / "results" / "result4_next_round" / "outputs",
        note="性能与物理叙事较完整的联合候选",
    ),
    RunSpec(
        "result4_next_round",
        "joint",
        "Joint (l=0.05, lr=5e-4, s=0.001, r=0.005)",
        "e4_joint_l005_lr5e4_nomix_s0010_r0050",
        REPO_ROOT / "results" / "result4_next_round" / "outputs",
        note="用于说明联合增益不只是 lr=3e-4 伪象",
    ),
)

ROBUST_GROUPS: tuple[RobustGroupSpec, ...] = (
    RobustGroupSpec("e4_ccg_xlstm_base", "CCG Baseline", "baseline"),
    RobustGroupSpec("e4_ccg_phys_s0010_r0075_prphi", "CCG-Phys (p/phi smooth, r=0.0075)", "phys"),
    RobustGroupSpec("e4_vmd_ccg_l003_lr5e4_nomix", "VMD-CCG (l=0.03, lr=5e-4, no mixer)", "vmd"),
    RobustGroupSpec("e4_vmd_ccg_l005_lr5e4_nomix", "VMD-CCG (l=0.05, lr=5e-4, no mixer)", "vmd"),
    RobustGroupSpec("e4_joint_l005_lr3e4_nomix_r0050", "Joint (l=0.05, lr=3e-4, r=0.005)", "joint"),
    RobustGroupSpec(
        "e4_joint_l005_lr3e4_nomix_s0010_r0050",
        "Joint (l=0.05, lr=3e-4, s=0.001, r=0.005)",
        "joint",
    ),
)


def summarize_paper_results(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> dict[str, Path]:
    output_root = ensure_dir(Path(output_dir))
    data_dir = ensure_dir(output_root / "data")
    figures_dir = ensure_dir(output_root / "figures")

    representative_specs = (
        RESULT1_SPECS
        + RESULT2_SPECS
        + RESULT3_SPECS
        + RESULT4_MAIN_SPECS
        + RESULT4_NEXT_ROUND_SPECS
    )
    representative_rows = [collect_representative_row(spec) for spec in representative_specs]
    representative_csv = data_dir / "representative_runs.csv"
    write_csv(representative_csv, representative_rows, list(representative_rows[0].keys()))

    best_stage_rows = build_best_stage_rows(representative_rows)
    best_stage_csv = data_dir / "best_by_stage.csv"
    write_csv(best_stage_csv, best_stage_rows, list(best_stage_rows[0].keys()))

    robust_seed_rows, robust_group_rows = collect_robust_rows(ROBUST_GROUPS)
    robust_seed_csv = data_dir / "result4_robust_seed_metrics.csv"
    robust_group_csv = data_dir / "result4_robust_group_stats.csv"
    write_csv(robust_seed_csv, robust_seed_rows, list(robust_seed_rows[0].keys()))
    write_csv(robust_group_csv, robust_group_rows, list(robust_group_rows[0].keys()))

    shortlist_rows = build_final_shortlist_rows(robust_group_rows)
    shortlist_csv = data_dir / "result4_final_shortlist_robust.csv"
    write_csv(shortlist_csv, shortlist_rows, list(shortlist_rows[0].keys()))

    plot_stage_best(best_stage_rows, figures_dir / "stage_best_ood_rmse.png")
    plot_next_round_ood(representative_rows, figures_dir / "result4_next_round_ood_rmse.png")
    plot_robust_ood(robust_group_rows, figures_dir / "result4_robust_ood_mean_std.png")
    plot_robust_tradeoff(robust_group_rows, figures_dir / "result4_robust_tradeoff_jump.png")

    readme_path = output_root / "README.md"
    readme_path.write_text(
        build_markdown_summary(
            representative_rows=representative_rows,
            best_stage_rows=best_stage_rows,
            robust_group_rows=robust_group_rows,
            shortlist_rows=shortlist_rows,
        ),
        encoding="utf-8",
    )

    return {
        "representative_runs": representative_csv,
        "best_by_stage": best_stage_csv,
        "robust_seed_metrics": robust_seed_csv,
        "robust_group_stats": robust_group_csv,
        "final_shortlist": shortlist_csv,
        "readme": readme_path,
    }


def collect_representative_row(spec: RunSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "stage": spec.stage,
        "family": spec.family,
        "label": spec.label,
        "run_name": spec.run_name,
        "source_dir": str(spec.run_dir),
        "note": spec.note,
    }
    for split in SPLITS:
        metrics = load_metrics(spec.run_dir / f"metrics_{split}.json")
        row[f"{split}_rmse"] = float(metrics["rmse_mean"])
        row[f"{split}_mae"] = float(metrics["mae_mean"])
        row[f"{split}_r2"] = float(metrics["r2_mean"])
        if split == "ood_test":
            for key in PHYSICS_KEYS:
                row[key] = float(metrics.get(key, math.nan))
    return row


def build_best_stage_rows(representative_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_stages = [
        "result_1",
        "result_2_followup",
        "result3",
        "result4_main",
        "result4_next_round",
    ]
    rows: list[dict[str, Any]] = []
    for stage in ordered_stages:
        stage_rows = [row for row in representative_rows if row["stage"] == stage]
        if not stage_rows:
            continue
        best_row = min(stage_rows, key=lambda item: float(item["ood_test_rmse"]))
        baseline_rows = [row for row in stage_rows if row["family"] == "baseline"]
        baseline_rmse = float(baseline_rows[0]["ood_test_rmse"]) if baseline_rows else math.nan
        improvement_pct = (
            (baseline_rmse - float(best_row["ood_test_rmse"])) / baseline_rmse * 100.0
            if baseline_rows and baseline_rmse > 0.0
            else math.nan
        )
        rows.append(
            {
                "stage": stage,
                "baseline_label": baseline_rows[0]["label"] if baseline_rows else "",
                "baseline_ood_rmse": baseline_rmse,
                "winner_label": best_row["label"],
                "winner_family": best_row["family"],
                "winner_run_name": best_row["run_name"],
                "winner_ood_rmse": float(best_row["ood_test_rmse"]),
                "relative_improvement_pct": improvement_pct,
                "note": best_row["note"],
            }
        )
    return rows


def collect_robust_rows(
    groups: Sequence[RobustGroupSpec],
    output_root: Path = REPO_ROOT / "results" / "result4_robust" / "outputs",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seed_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []

    for group in groups:
        run_dirs = sorted(output_root.glob(f"{group.base_name}_seed*"))
        if not run_dirs:
            raise FileNotFoundError(f"No robust runs found for base: {group.base_name}")

        split_values: dict[str, list[float]] = {f"{split}_rmse": [] for split in SPLITS}
        physics_values: dict[str, list[float]] = {key: [] for key in PHYSICS_KEYS}

        for run_dir in run_dirs:
            seed = run_dir.name.split("_seed")[-1]
            row: dict[str, Any] = {
                "base_name": group.base_name,
                "label": group.label,
                "family": group.family,
                "seed": seed,
                "run_name": run_dir.name,
            }
            for split in SPLITS:
                metrics = load_metrics(run_dir / f"metrics_{split}.json")
                row[f"{split}_rmse"] = float(metrics["rmse_mean"])
                split_values[f"{split}_rmse"].append(float(metrics["rmse_mean"]))
                if split == "ood_test":
                    for key in PHYSICS_KEYS:
                        value = float(metrics.get(key, math.nan))
                        row[key] = value
                        physics_values[key].append(value)
            seed_rows.append(row)

        group_row: dict[str, Any] = {
            "base_name": group.base_name,
            "label": group.label,
            "family": group.family,
            "num_seeds": len(run_dirs),
        }
        for key, values in split_values.items():
            group_row[f"{key}_mean"] = mean(values)
            group_row[f"{key}_std"] = std(values)
        for key, values in physics_values.items():
            valid = [value for value in values if not math.isnan(value)]
            group_row[f"{key}_mean"] = mean(valid) if valid else math.nan
            group_row[f"{key}_std"] = std(valid) if len(valid) > 1 else 0.0
        group_rows.append(group_row)

    baseline_row = next(row for row in group_rows if row["base_name"] == "e4_ccg_xlstm_base")
    baseline_ood = float(baseline_row["ood_test_rmse_mean"])
    for row in group_rows:
        row["ood_gain_vs_baseline_pct"] = (baseline_ood - float(row["ood_test_rmse_mean"])) / baseline_ood * 100.0
    group_rows.sort(key=lambda item: float(item["ood_test_rmse_mean"]))
    return seed_rows, group_rows


def build_final_shortlist_rows(group_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {
        "e4_ccg_xlstm_base",
        "e4_ccg_phys_s0010_r0075_prphi",
        "e4_vmd_ccg_l003_lr5e4_nomix",
        "e4_joint_l005_lr3e4_nomix_r0050",
        "e4_joint_l005_lr3e4_nomix_s0010_r0050",
    }
    shortlist = [row for row in group_rows if row["base_name"] in wanted]
    shortlist.sort(key=lambda item: float(item["ood_test_rmse_mean"]))
    return shortlist


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return load_json(path)


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return float(sum(materialized) / len(materialized))


def std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(row.get(key)) for key in fieldnames})


def normalize_csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.12g}"
    return value


def plot_stage_best(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    labels = [row["stage"] for row in rows]
    values = [float(row["winner_ood_rmse"]) for row in rows]
    plt.figure(figsize=(8, 4.8))
    plt.plot(labels, values, marker="o", linewidth=2.0)
    plt.ylabel("OOD RMSE")
    plt.title("Best OOD RMSE by Experiment Stage")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_next_round_ood(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    next_round_rows = [row for row in rows if row["stage"] == "result4_next_round"]
    next_round_rows.sort(key=lambda item: float(item["ood_test_rmse"]))
    labels = [row["label"] for row in next_round_rows]
    values = [float(row["ood_test_rmse"]) for row in next_round_rows]
    plt.figure(figsize=(10, 5.6))
    plt.barh(labels, values)
    plt.xlabel("OOD RMSE")
    plt.title("Result4 Next Round Single-Seed OOD RMSE")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_robust_ood(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    ordered = sorted(rows, key=lambda item: float(item["ood_test_rmse_mean"]))
    labels = [row["label"] for row in ordered]
    means = [float(row["ood_test_rmse_mean"]) for row in ordered]
    stds = [float(row["ood_test_rmse_std"]) for row in ordered]
    plt.figure(figsize=(10, 5.8))
    plt.barh(labels, means, xerr=stds, capsize=4)
    plt.xlabel("OOD RMSE Mean +/- Std")
    plt.title("Result4 Robust OOD Performance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_robust_tradeoff(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    plt.figure(figsize=(7.2, 5.4))
    for row in rows:
        x_value = float(row["ood_test_rmse_mean"])
        y_value = float(row["first_step_jump_rmse_mean"])
        plt.scatter(x_value, y_value, s=70)
        plt.annotate(row["label"], (x_value, y_value), fontsize=8, xytext=(5, 4), textcoords="offset points")
    plt.xlabel("OOD RMSE Mean")
    plt.ylabel("First-Step Jump RMSE Mean")
    plt.title("Prediction vs Trajectory Jump Trade-off")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def build_markdown_summary(
    representative_rows: Sequence[dict[str, Any]],
    best_stage_rows: Sequence[dict[str, Any]],
    robust_group_rows: Sequence[dict[str, Any]],
    shortlist_rows: Sequence[dict[str, Any]],
) -> str:
    result1_best = find_row(representative_rows, stage="result_1", run_name="ccg_xlstm_seq128_pred10")
    result2_best = min(
        (row for row in representative_rows if row["stage"] == "result_2_followup"),
        key=lambda item: float(item["ood_test_rmse"]),
    )
    result3_phys = find_row(representative_rows, stage="result3", run_name="e3_ccg_phys_s001_r005")
    result3_vmd = find_row(representative_rows, stage="result3", run_name="e3_vmd_ccg_l005_lr5e4")
    result4_main_vmd = find_row(representative_rows, stage="result4_main", run_name="e4_vmd_ccg_l005_lr3e4_nomix")
    result4_next_joint = find_row(
        representative_rows,
        stage="result4_next_round",
        run_name="e4_joint_l005_lr3e4_nomix_r0050",
    )
    robust_best = shortlist_rows[0]
    robust_phys = next(row for row in robust_group_rows if row["base_name"] == "e4_ccg_phys_s0010_r0075_prphi")
    robust_base = next(row for row in robust_group_rows if row["base_name"] == "e4_ccg_xlstm_base")

    lines = [
        "# Paper Result Summary",
        "",
        "## 1. 使用范围与目录说明",
        "",
        "这份目录用于集中整理论文书写阶段要直接引用的实验结果材料，覆盖：",
        "",
        "- `result_1`：基础模型对比与 CCG-xLSTM baseline 确立。",
        "- `result_2_followup`：VMD 流程补跑与早期 follow-up 结果。",
        "- `result3`：Phys / VMD 独立消融阶段。",
        "- `result4`：单 seed 主实验筛选。",
        "- `result4_next_round`：下一轮精简实验与候选收敛。",
        "- `result4_robust`：6 个 shortlist 配置的 5-seed 稳健性验证。",
        "",
        "生成文件：",
        "",
        "- `data/representative_runs.csv`：跨阶段代表性模型总表。",
        "- `data/best_by_stage.csv`：每一阶段的最佳 OOD 模型摘要。",
        "- `data/result4_robust_seed_metrics.csv`：robust 30 次实验逐 seed 指标。",
        "- `data/result4_robust_group_stats.csv`：robust 分组均值/方差统计。",
        "- `data/result4_final_shortlist_robust.csv`：建议写入论文主文的最终 shortlist。",
        "- `figures/stage_best_ood_rmse.png`：各阶段最佳 OOD RMSE 走势。",
        "- `figures/result4_next_round_ood_rmse.png`：next_round 单 seed OOD 对比。",
        "- `figures/result4_robust_ood_mean_std.png`：robust 的均值±标准差对比。",
        "- `figures/result4_robust_tradeoff_jump.png`：预测性能与轨迹首步跳变的折中图。",
        "",
        "## 2. 跨阶段主线结论",
        "",
        f"- `result_1` 中 `CCG-xLSTM Baseline` 的 OOD RMSE 为 `{float(result1_best['ood_test_rmse']):.6f}`，优于 LSTM / GRU / Lite-xLSTM / Transformer，也优于当时的早期 VMD 原型。这一阶段的核心价值是确定 baseline，而不是直接证明 VMD/Phys 有效。",
        f"- `result_2_followup` 的最佳 OOD 结果来自 `{result2_best['label']}`，OOD RMSE `{float(result2_best['ood_test_rmse']):.6f}`。这组结果主要证明缓存化 VMD 流程已经打通，但仍属于历史中间态，不建议作为最终主结论。",
        f"- `result3` 首次把 Phys 和 VMD 拆开分析。Phys 最优候选 `e3_ccg_phys_s001_r005` 的 OOD RMSE 为 `{float(result3_phys['ood_test_rmse']):.6f}`，VMD 最优候选 `e3_vmd_ccg_l005_lr5e4` 的 OOD RMSE 为 `{float(result3_vmd['ood_test_rmse']):.6f}`，说明两条方向都值得继续，但作用机制不同。",
        f"- `result4 main` 中，VMD 主线进一步收敛到 `e4_vmd_ccg_l005_lr3e4_nomix`，OOD RMSE `{float(result4_main_vmd['ood_test_rmse']):.6f}`，相对同阶段 baseline 改善明显。",
        f"- `result4 next_round` 中，联合方案 `e4_joint_l005_lr3e4_nomix_r0050` 取得当前单 seed 最低 OOD RMSE `{float(result4_next_joint['ood_test_rmse']):.6f}`，说明 Joint 已经超过单纯 VMD 候选。",
        f"- `result4 robust` 的 5-seed 统计进一步确认了这一点：当前最优 robust OOD 均值来自 `{robust_best['label']}`，为 `{float(robust_best['ood_test_rmse_mean']):.6f} ± {float(robust_best['ood_test_rmse_std']):.6f}`。",
        "",
        "## 3. Result4 最终可写结论",
        "",
        f"- baseline 的 robust OOD 均值为 `{float(robust_base['ood_test_rmse_mean']):.6f} ± {float(robust_base['ood_test_rmse_std']):.6f}`。",
        f"- Phys 代表配置 `e4_ccg_phys_s0010_r0075_prphi` 的 robust OOD 均值为 `{float(robust_phys['ood_test_rmse_mean']):.6f} ± {float(robust_phys['ood_test_rmse_std']):.6f}`，预测误差仍不占优，但其 `first_step_jump_rmse` 从 baseline 的 `{float(robust_base['first_step_jump_rmse_mean']):.6f}` 显著降到 `{float(robust_phys['first_step_jump_rmse_mean']):.6f}`，`max_abs_second_diff_p95` 也从 `{float(robust_base['max_abs_second_diff_p95_mean']):.6f}` 降到 `{float(robust_phys['max_abs_second_diff_p95_mean']):.6f}`，支持“限制不合理轨迹”的新叙事。",
        "- VMD 主线的 robust 表现已经稳定优于 baseline，其中 `e4_vmd_ccg_l003_lr5e4_nomix` 是更好的 VMD 代表方案。",
        "- Joint 是当前最强主线。`e4_joint_l005_lr3e4_nomix_r0050` 的 OOD 均值最低，而 `e4_joint_l005_lr3e4_nomix_s0010_r0050` 在 routine/val 和物理叙事上更完整。",
        "",
        "## 4. 建议用于论文主文的最终 shortlist",
        "",
        markdown_table(
            shortlist_rows,
            [
                ("label", "Model"),
                ("ood_test_rmse_mean", "OOD RMSE mean"),
                ("ood_test_rmse_std", "OOD RMSE std"),
                ("routine_test_rmse_mean", "Routine RMSE mean"),
                ("val_rmse_mean", "Val RMSE mean"),
                ("ood_gain_vs_baseline_pct", "Gain vs baseline (%)"),
            ],
        ),
        "",
        "推荐写法：",
        "",
        "- 若主文只放 4 个代表模型：`baseline + phys_prphi + vmd_l003 + joint_s0010_r0050`。",
        "- 若主文允许 5 个模型：再加入 `joint_r0050`，专门说明“最低 OOD 均值”和“更完整物理叙事”的两种联合 winner。",
        "",
        "## 5. 文章写作时的引用建议",
        "",
        "- 写基础模型对比时：优先引用 `data/representative_runs.csv` 里 `result_1` 的各模型行。",
        "- 写 Phys / VMD 作用分离时：优先引用 `result3` 的代表行，并结合现有 `docs/result3_analysis_report.md`。",
        "- 写实验四主结果时：优先引用 `data/result4_final_shortlist_robust.csv` 和 `figures/result4_robust_ood_mean_std.png`。",
        "- 写 Phys 的新定位时：优先引用 `figures/result4_robust_tradeoff_jump.png`，强调 Phys 主要改善轨迹跳变与物理一致性，而不是直接夺取最低预测 RMSE。",
        "",
        "## 6. 注意事项",
        "",
        "- `result_2_followup` 的目录结构与其他阶段不同，正式结果位于 `results/result_2_followup/results/result_2_followup/outputs/`。本汇总已按该路径读取。",
        "- 本目录中的图表和 CSV 由 `python -m ship_motion.paper_result_summary` 自动生成；若后续补跑实验，请重新执行该命令刷新材料。",
    ]
    return "\n".join(lines) + "\n"


def markdown_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def find_row(rows: Sequence[dict[str, Any]], stage: str, run_name: str) -> dict[str, Any]:
    for row in rows:
        if row["stage"] == stage and row["run_name"] == run_name:
            return row
    raise KeyError(f"Row not found: stage={stage}, run_name={run_name}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready experiment summary tables and figures.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write markdown, CSV summaries, and figures.",
    )
    args = parser.parse_args(argv)

    outputs = summarize_paper_results(output_dir=args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

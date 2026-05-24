from __future__ import annotations

"""Step 07：实验结果绘图脚本。"""

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ship_motion.utils import ensure_dir, load_yaml


DEFAULT_ABLATION_LIST = Path(__file__).resolve().parents[2] / "configs" / "ablation_list.yaml"
TARGET_NAME_TO_INDEX = {"u": 0, "v": 1, "p": 2, "r": 3, "phi": 4}


def generate_plots(
    ablation_config_path: str | Path = DEFAULT_ABLATION_LIST,
    runs: Sequence[str] | None = None,
    run_output_root: str | Path = "outputs",
    summary_output_dir: str | Path = "outputs/summary",
    smoke: bool = False,
    num_windows: int = 200,
    horizon: int = 0,
) -> list[Path]:
    """生成预测曲线图和柱状图。"""
    ablation_cfg = load_yaml(ablation_config_path)
    selected_runs = select_runs(ablation_cfg, runs)
    summary_dir = ensure_dir(Path(summary_output_dir))
    figures_dir = ensure_dir(summary_dir / "figures")

    figure_paths: list[Path] = []

    # 预测曲线图：默认画 u 和 phi，在 routine / OOD 两个 split 上各一组。
    # 这里优先保留论文主线中最有代表性的模型，避免一张图里线条过多难以阅读。
    curve_run_names = [
        "persistence_seq128_pred10",
        "lstm_seq128_pred10",
        "transformer_seq128_pred10",
        "lite_xlstm_seq128_pred10",
        "ccg_xlstm_seq128_pred10",
        "vmd_ccg_xlstm_seq128_pred10",
        "vmd_ccg_phys_xlstm_seq128_pred10",
    ]
    available_run_names = {str(item["run_name"]) for item in selected_runs}
    curve_runs = [item for item in curve_run_names if item in available_run_names]

    for split_name in ["routine_test", "ood_test"]:
        for target_name in ["u", "phi"]:
            figure_path = figures_dir / f"pred_{target_name}_{split_name.replace('_test', '')}.png"
            plot_prediction_curves(
                ablation_cfg=ablation_cfg,
                run_names=curve_runs,
                run_output_root=run_output_root,
                figure_path=figure_path,
                split_name=split_name,
                target_name=target_name,
                smoke=smoke,
                num_windows=num_windows,
                horizon=horizon,
            )
            figure_paths.append(figure_path)

    routine_csv = summary_dir / "ablation_routine_test.csv"
    ood_csv = summary_dir / "ablation_ood_test.csv"
    physics_csv = summary_dir / "physics_metrics.csv"
    rmse_bar_path = figures_dir / "rmse_bar.png"
    rmse_ood_bar_path = figures_dir / "rmse_ood_bar.png"
    roll_bar_path = figures_dir / "roll_consistency_bar.png"

    plot_bar_chart(
        csv_path=routine_csv,
        figure_path=rmse_bar_path,
        metric_key="rmse_mean",
        title="Routine Test RMSE Mean",
        ylabel="RMSE Mean",
    )
    figure_paths.append(rmse_bar_path)

    plot_bar_chart(
        csv_path=ood_csv,
        figure_path=rmse_ood_bar_path,
        metric_key="rmse_mean",
        title="OOD Test RMSE Mean",
        ylabel="RMSE Mean",
    )
    figure_paths.append(rmse_ood_bar_path)

    plot_bar_chart(
        csv_path=physics_csv,
        figure_path=roll_bar_path,
        metric_key="roll_consistency_rmse",
        title="Roll Consistency RMSE",
        ylabel="Roll Consistency RMSE",
        split_filter="ood_test",
    )
    figure_paths.append(roll_bar_path)
    return figure_paths


def plot_prediction_curves(
    ablation_cfg: dict[str, Any],
    run_names: Sequence[str],
    run_output_root: str | Path,
    figure_path: str | Path,
    split_name: str,
    target_name: str,
    smoke: bool,
    num_windows: int,
    horizon: int,
) -> None:
    """画某个变量在某个 split 上的预测曲线。"""
    index = TARGET_NAME_TO_INDEX[target_name]
    run_root = Path(run_output_root)
    plt.figure(figsize=(12, 6))
    true_drawn = False

    for run_name in run_names:
        run_dir = run_root / resolve_run_name(run_name, smoke=smoke)
        prediction_path = run_dir / f"predictions_{split_name}.npz"
        if not prediction_path.exists():
            continue
        data = np.load(prediction_path)
        y_pred = data["y_pred"]
        y_true = data["y_true"]
        limit = min(num_windows, y_pred.shape[0])
        x_axis = np.arange(limit)

        if not true_drawn:
            plt.plot(x_axis, y_true[:limit, horizon, index], label="True", linewidth=2.0, color="black")
            true_drawn = True

        display_name = resolve_display_name(ablation_cfg, run_name)
        plt.plot(x_axis, y_pred[:limit, horizon, index], label=display_name, linewidth=1.4)

    plt.title(f"{target_name} prediction curves ({split_name}, horizon={horizon + 1})")
    plt.xlabel("Window index")
    plt.ylabel(target_name)
    plt.legend()
    plt.tight_layout()
    target = Path(figure_path)
    ensure_dir(target.parent)
    plt.savefig(target, dpi=150)
    plt.close()


def plot_bar_chart(
    csv_path: str | Path,
    figure_path: str | Path,
    metric_key: str,
    title: str,
    ylabel: str,
    split_filter: str | None = None,
) -> None:
    """根据汇总 CSV 画简单柱状图。"""
    rows = read_csv_rows(csv_path)
    if split_filter is not None:
        rows = [row for row in rows if row.get("split") == split_filter]
    if not rows:
        raise ValueError(f"No rows found for plotting from {csv_path}")

    rows = sorted(rows, key=lambda row: int(row.get("order", "999")))
    labels = [row["model"] for row in rows]
    values = [float(row[metric_key]) for row in rows]

    plt.figure(figsize=(12, 6))
    plt.bar(labels, values)
    plt.xticks(rotation=30, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    target = Path(figure_path)
    ensure_dir(target.parent)
    plt.savefig(target, dpi=150)
    plt.close()


def select_runs(ablation_cfg: dict[str, Any], runs: Sequence[str] | None) -> list[dict[str, Any]]:
    items = list(ablation_cfg.get("runs", []))
    if not runs:
        return items
    wanted = set(runs)
    selected = [item for item in items if str(item["run_name"]) in wanted]
    missing = wanted - {str(item["run_name"]) for item in selected}
    if missing:
        raise ValueError(f"Runs not found in ablation list: {sorted(missing)}")
    return selected


def resolve_run_name(run_name: str, smoke: bool) -> str:
    return f"{run_name}_smoke" if smoke and not run_name.endswith("_smoke") else run_name


def resolve_display_name(ablation_cfg: dict[str, Any], run_name: str) -> str:
    for item in ablation_cfg.get("runs", []):
        if str(item["run_name"]) == run_name:
            return str(item.get("display_name", run_name))
    return run_name


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_runs(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot summarized results for Step 07.")
    parser.add_argument("--ablation-config", default=str(DEFAULT_ABLATION_LIST))
    parser.add_argument("--runs", default=None, help="Comma-separated run names to plot.")
    parser.add_argument("--run-output-root", default="outputs")
    parser.add_argument("--summary-output-dir", default="outputs/summary")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num-windows", type=int, default=200)
    parser.add_argument("--horizon", type=int, default=0)
    args = parser.parse_args(argv)

    figure_paths = generate_plots(
        ablation_config_path=args.ablation_config,
        runs=parse_runs(args.runs),
        run_output_root=args.run_output_root,
        summary_output_dir=args.summary_output_dir,
        smoke=args.smoke,
        num_windows=args.num_windows,
        horizon=args.horizon,
    )
    for path in figure_paths:
        print(path)


if __name__ == "__main__":
    main()

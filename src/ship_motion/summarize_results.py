from __future__ import annotations

"""Step 07：实验结果汇总脚本。"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Sequence

from ship_motion.utils import ensure_dir, load_json, load_yaml


DEFAULT_ABLATION_LIST = Path(__file__).resolve().parents[2] / "configs" / "ablation_list.yaml"


def summarize_results(
    ablation_config_path: str | Path = DEFAULT_ABLATION_LIST,
    runs: Sequence[str] | None = None,
    run_output_root: str | Path = "outputs",
    summary_output_dir: str | Path = "outputs/summary",
    smoke: bool = False,
    skip_missing_runs: bool = True,
) -> dict[str, Path]:
    """汇总 routine / OOD / physics 指标表。"""
    ablation_cfg = load_yaml(ablation_config_path)
    selected_runs = select_runs(ablation_cfg, runs)
    run_root = Path(run_output_root)
    summary_dir = ensure_dir(Path(summary_output_dir))

    routine_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    skipped_runs: list[str] = []

    for item in selected_runs:
        run_name = resolve_run_name(str(item["run_name"]), smoke=smoke)
        run_dir = run_root / run_name
        if not run_dir.exists():
            message = f"Run directory not found: {run_dir}"
            if not skip_missing_runs:
                raise FileNotFoundError(message)
            print(f"[Warn] {message}; skipping this run.", file=sys.stderr)
            skipped_runs.append(run_name)
            continue

        display_name = str(item.get("display_name", run_name))
        order = int(item.get("order", 999))
        category = str(item.get("category", "main"))

        required_metric_paths: dict[str, Path] = {
            split_name: run_dir / f"metrics_{split_name}.json"
            for split_name in ("routine_test", "ood_test")
        }
        missing_metric_path = next((path for path in required_metric_paths.values() if not path.exists()), None)
        if missing_metric_path is not None:
            message = f"Metrics file not found: {missing_metric_path}"
            if not skip_missing_runs:
                raise FileNotFoundError(message)
            print(f"[Warn] {message}; skipping this run.", file=sys.stderr)
            skipped_runs.append(run_name)
            continue

        for split_name, collector in [("routine_test", routine_rows), ("ood_test", ood_rows)]:
            metrics = load_json(required_metric_paths[split_name])
            collector.append(build_metric_row(order, display_name, run_name, category, split_name, metrics))

            if "smoothness" in metrics or "roll_consistency_rmse" in metrics:
                physics_rows.append(
                    {
                        "order": order,
                        "model": display_name,
                        "run_name": run_name,
                        "split": split_name,
                        "smoothness": float(metrics.get("smoothness", 0.0)),
                        "roll_consistency_rmse": float(metrics.get("roll_consistency_rmse", 0.0)),
                        "first_step_jump_rmse": float(metrics.get("first_step_jump_rmse", 0.0)),
                        "max_abs_second_diff_p95": float(metrics.get("max_abs_second_diff_p95", 0.0)),
                    }
                )

    if not routine_rows or not ood_rows:
        raise FileNotFoundError(
            f"No complete results were found under {run_root}. "
            "Please check whether the target run directories and metrics JSON files exist."
        )

    routine_path = summary_dir / "ablation_routine_test.csv"
    ood_path = summary_dir / "ablation_ood_test.csv"
    physics_path = summary_dir / "physics_metrics.csv"

    write_csv(
        routine_path,
        routine_rows,
        [
            "order",
            "model",
            "run_name",
            "category",
            "split",
            "mae_mean",
            "rmse_mean",
            "r2_mean",
            "rmse_u",
            "rmse_v",
            "rmse_p",
            "rmse_r",
            "rmse_phi",
        ],
    )
    write_csv(
        ood_path,
        ood_rows,
        [
            "order",
            "model",
            "run_name",
            "category",
            "split",
            "mae_mean",
            "rmse_mean",
            "r2_mean",
            "rmse_u",
            "rmse_v",
            "rmse_p",
            "rmse_r",
            "rmse_phi",
        ],
    )
    write_csv(
        physics_path,
        physics_rows,
        [
            "order",
            "model",
            "run_name",
            "split",
            "smoothness",
            "roll_consistency_rmse",
            "first_step_jump_rmse",
            "max_abs_second_diff_p95",
        ],
    )
    if skipped_runs:
        print(
            "[Warn] Skipped runs due to missing directories or metrics: "
            + ", ".join(sorted(set(skipped_runs))),
            file=sys.stderr,
        )
    return {
        "routine": routine_path,
        "ood": ood_path,
        "physics": physics_path,
    }


def build_metric_row(
    order: int,
    display_name: str,
    run_name: str,
    category: str,
    split_name: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """提取论文主表常用字段。"""
    return {
        "order": order,
        "model": display_name,
        "run_name": run_name,
        "category": category,
        "split": split_name,
        "mae_mean": float(metrics["mae_mean"]),
        "rmse_mean": float(metrics["rmse_mean"]),
        "r2_mean": float(metrics["r2_mean"]),
        "rmse_u": float(metrics.get("rmse_u", 0.0)),
        "rmse_v": float(metrics.get("rmse_v", 0.0)),
        "rmse_p": float(metrics.get("rmse_p", 0.0)),
        "rmse_r": float(metrics.get("rmse_r", 0.0)),
        "rmse_phi": float(metrics.get("rmse_phi", 0.0)),
    }


def select_runs(ablation_cfg: dict[str, Any], runs: Sequence[str] | None) -> list[dict[str, Any]]:
    """从 ablation list 中筛选要汇总的 run。"""
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
    """smoke 模式下自动读取 *_smoke 目录。"""
    return f"{run_name}_smoke" if smoke and not run_name.endswith("_smoke") else run_name


def write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    """把汇总结果写成 CSV。"""
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item.get("order", 999)), str(item.get("model", "")))):
            writer.writerow(row)


def parse_runs(text: str | None) -> list[str] | None:
    if text is None or not text.strip():
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize ablation results for Step 07.")
    parser.add_argument("--ablation-config", default=str(DEFAULT_ABLATION_LIST))
    parser.add_argument("--runs", default=None, help="Comma-separated run names to summarize.")
    parser.add_argument("--run-output-root", default="outputs")
    parser.add_argument("--summary-output-dir", default="outputs/summary")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--strict-missing-runs",
        action="store_true",
        help="Fail immediately when a run directory or metrics file is missing.",
    )
    args = parser.parse_args(argv)

    outputs = summarize_results(
        ablation_config_path=args.ablation_config,
        runs=parse_runs(args.runs),
        run_output_root=args.run_output_root,
        summary_output_dir=args.summary_output_dir,
        smoke=args.smoke,
        skip_missing_runs=not args.strict_missing_runs,
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()

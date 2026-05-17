from __future__ import annotations

"""Step 02 训练入口：常规基线模型 + 统一训练评估框架。"""

import argparse
import copy
import csv
import time
from pathlib import Path
from typing import Any, Sequence

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from ship_motion.data.dataset import DEFAULT_CONFIG_PATH, build_datasets
from ship_motion.evaluate import (
    apply_vmd_smoke_overrides,
    build_loaders,
    build_runtime_model_config,
    compute_model_losses,
    default_run_dir,
    evaluate_model,
    move_batch_to_device,
    resolve_device,
)
from ship_motion.models import build_model_from_config
from ship_motion.utils import ensure_dir, load_yaml, save_json, set_seed


def build_model(config: dict[str, Any]) -> nn.Module:
    """根据当前配置构造模型。"""
    model_cfg = build_runtime_model_config(config)
    return build_model_from_config(model_cfg)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Any | None = None,
    y_std: Sequence[float] | None = None,
    lambda_vmd: float = 0.0,
    physics_cfg: dict[str, Any] | None = None,
    grad_clip: float | None = None,
    max_steps: int | None = None,
) -> float:
    """执行一个 epoch 的标准化空间 MSE 训练。"""
    criterion = nn.MSELoss()
    model.train()
    losses: list[float] = []

    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch["x"], batch=batch)
        _, loss, _, _ = compute_model_losses(
            model_output=output,
            batch=batch,
            criterion=criterion,
            scaler=scaler,
            y_std=y_std,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
        )
        loss.backward()

        # 梯度裁剪对 RNN 类模型尤其有帮助，可以降低梯度爆炸风险。
        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        losses.append(float(loss.item()))

    if not losses:
        raise ValueError("Training loader produced no batches.")
    return float(sum(losses) / len(losses))


def validate(
    model: nn.Module,
    loader: DataLoader,
    scaler: Any,
    target_cols: Sequence[str],
    device: torch.device,
    lambda_vmd: float = 0.0,
    physics_cfg: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> tuple[dict[str, float], float]:
    """验证集评估，指标在真实物理尺度上计算。"""
    return evaluate_model(
        model=model,
        loader=loader,
        scaler=scaler,
        target_cols=target_cols,
        device=device,
        lambda_vmd=lambda_vmd,
        physics_cfg=physics_cfg,
        max_steps=max_steps,
    )


def fit(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """训练并评估一个 Step 02 基线模型。"""
    runtime_config = copy.deepcopy(config)
    if smoke:
        apply_smoke_overrides(runtime_config)

    set_seed(int(runtime_config.get("train", {}).get("seed", 42)))
    bundle = build_datasets(runtime_config, smoke=False)
    train_cfg = runtime_config.get("train", {})
    data_cfg = runtime_config.get("data", {})
    run_dir = ensure_dir(default_run_dir(runtime_config))

    # 保存“本次真正运行时使用的配置”和 scaler，
    # 后续复现实验或单独评估时会很有用。
    save_runtime_config(runtime_config, run_dir / "config.yaml")
    bundle["scaler"].save(run_dir / "scaler.json")

    loaders = build_loaders(
        bundle=bundle,
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    device = resolve_device(str(train_cfg.get("device", "auto")))
    model = build_model(runtime_config).to(device)
    optimizer = build_optimizer(model, train_cfg)
    max_eval_steps = train_cfg.get("max_eval_steps")
    lambda_vmd = float(runtime_config.get("vmd", {}).get("lambda_vmd", 0.0))
    physics_cfg = runtime_config.get("physics", {})

    # Persistence 没有可训练参数，会走这个分支：
    # 不训练，只直接评估并输出统一格式文件。
    if optimizer is None:
        val_metrics, val_loss = validate(
            model=model,
            loader=loaders["val"],
            scaler=bundle["scaler"],
            target_cols=data_cfg["target_cols"],
            device=device,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            max_steps=max_eval_steps,
        )
        save_checkpoint(run_dir / "best.pt", model, runtime_config, best_metric=val_metrics["rmse_mean"], epoch=0)
        write_train_log(
            run_dir / "train_log.csv",
            rows=[
                {
                    "epoch": 0,
                    "train_loss": 0.0,
                    "val_loss": val_loss,
                    "val_rmse_mean": val_metrics["rmse_mean"],
                    "elapsed_sec": 0.0,
                }
            ],
        )
        all_metrics = finalize_and_evaluate(
            model=model,
            bundle=bundle,
            loaders=loaders,
            config=runtime_config,
            run_dir=run_dir,
            device=device,
        )
        all_metrics["best_epoch"] = 0
        all_metrics["best_val_rmse_mean"] = val_metrics["rmse_mean"]
        return all_metrics

    history: list[dict[str, float]] = []
    best_state_dict = copy.deepcopy(model.state_dict())
    best_val_rmse = float("inf")
    best_epoch = 0
    patience = int(train_cfg.get("early_stop_patience", 8))
    epochs = int(train_cfg.get("epochs", 1))
    grad_clip = train_cfg.get("grad_clip")
    max_train_steps = train_cfg.get("max_train_steps_per_epoch")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        start_time = time.perf_counter()
        train_loss = train_one_epoch(
            model=model,
            loader=loaders["train"],
            optimizer=optimizer,
            device=device,
            scaler=bundle["scaler"],
            y_std=bundle["scaler"].y_std,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            grad_clip=float(grad_clip) if grad_clip is not None else None,
            max_steps=max_train_steps,
        )
        val_metrics, val_loss = validate(
            model=model,
            loader=loaders["val"],
            scaler=bundle["scaler"],
            target_cols=data_cfg["target_cols"],
            device=device,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            max_steps=max_eval_steps,
        )
        elapsed = time.perf_counter() - start_time
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_rmse_mean": val_metrics["rmse_mean"],
                "elapsed_sec": elapsed,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_rmse_mean={val_metrics['rmse_mean']:.6f}"
        )

        # Step 02 的早停依据是 val_rmse_mean，越低越好。
        if val_metrics["rmse_mean"] < best_val_rmse:
            best_val_rmse = float(val_metrics["rmse_mean"])
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            save_checkpoint(
                run_dir / "best.pt",
                model,
                runtime_config,
                best_metric=best_val_rmse,
                epoch=epoch,
            )
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    # 评估前先恢复最佳权重，而不是最后一轮权重。
    model.load_state_dict(best_state_dict)
    write_train_log(run_dir / "train_log.csv", history)
    all_metrics = finalize_and_evaluate(
        model=model,
        bundle=bundle,
        loaders=loaders,
        config=runtime_config,
        run_dir=run_dir,
        device=device,
    )
    all_metrics["best_epoch"] = best_epoch
    all_metrics["best_val_rmse_mean"] = best_val_rmse
    return all_metrics


def finalize_and_evaluate(
    model: nn.Module,
    bundle: dict[str, Any],
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """保存 val / routine test / OOD test 指标。"""
    data_cfg = config["data"]
    max_eval_steps = config.get("train", {}).get("max_eval_steps")
    results: dict[str, Any] = {}
    for split in ["val", "routine_test", "ood_test"]:
        metrics, _ = evaluate_model(
            model=model,
            loader=loaders[split],
            scaler=bundle["scaler"],
            target_cols=data_cfg["target_cols"],
            device=device,
            lambda_vmd=float(config.get("vmd", {}).get("lambda_vmd", 0.0)),
            physics_cfg=config.get("physics", {}),
            max_steps=max_eval_steps,
        )
        save_json(metrics, run_dir / f"metrics_{split}.json")
        results[split] = metrics
    return results


def build_optimizer(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer | None:
    """对可训练模型创建 AdamW 优化器；无参数模型返回 None。"""
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not params:
        return None
    return torch.optim.AdamW(
        params,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    config: dict[str, Any],
    best_metric: float,
    epoch: int,
) -> None:
    """保存当前最佳模型权重。"""
    target = Path(path)
    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "run_name": config.get("run_name"),
            "model_name": config.get("model", {}).get("name"),
        },
        target,
    )


def save_runtime_config(config: dict[str, Any], path: str | Path) -> None:
    """保存当前运行真正使用的配置。"""
    serializable = copy.deepcopy(config)
    serializable.pop("__config_path__", None)
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(serializable, f, allow_unicode=True, sort_keys=False)


def write_train_log(path: str | Path, rows: list[dict[str, float]]) -> None:
    """将训练日志写成 CSV，方便后续画图或汇总。"""
    fieldnames = ["epoch", "train_loss", "val_loss", "val_rmse_mean", "elapsed_sec"]
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def apply_smoke_overrides(config: dict[str, Any]) -> None:
    """把正式配置改成极轻量 smoke 配置。"""
    config["run_name"] = f"{config.get('run_name', 'run')}_smoke"
    config["output_dir"] = "outputs/smoke"

    data_cfg = config.setdefault("data", {})
    data_cfg["max_files"] = min(int(data_cfg.get("max_files") or 1), 1)
    data_cfg["max_windows_per_file"] = min(int(data_cfg.get("max_windows_per_file") or 16), 16)

    train_cfg = config.setdefault("train", {})
    train_cfg["batch_size"] = min(int(train_cfg.get("batch_size", 8)), 8)
    train_cfg["epochs"] = min(int(train_cfg.get("epochs", 1)), 1)
    train_cfg["early_stop_patience"] = 1
    train_cfg["num_workers"] = 0
    train_cfg["max_train_steps_per_epoch"] = 2
    train_cfg["max_eval_steps"] = 2
    apply_vmd_smoke_overrides(config)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Step 02 baseline models.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    summary = fit(config, smoke=args.smoke)
    printable = {
        "best_epoch": summary.get("best_epoch"),
        "best_val_rmse_mean": summary.get("best_val_rmse_mean"),
        "val_rmse_mean": summary["val"]["rmse_mean"],
        "routine_test_rmse_mean": summary["routine_test"]["rmse_mean"],
        "ood_test_rmse_mean": summary["ood_test"]["rmse_mean"],
    }
    print(yaml.safe_dump(printable, allow_unicode=True, sort_keys=False))


if __name__ == "__main__":
    main()

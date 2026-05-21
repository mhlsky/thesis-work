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
from tqdm.auto import tqdm

from ship_motion.data.dataset import DEFAULT_CONFIG_PATH, build_datasets
from ship_motion.evaluate import (
    apply_vmd_smoke_overrides,
    build_loaders,
    build_runtime_model_config,
    compute_model_losses,
    default_run_dir,
    evaluate_model,
    evaluate_model_with_predictions,
    log_runtime_environment,
    move_batch_to_device,
    progress_is_interactive,
    resolve_loader_settings,
    resolve_progress_total,
    resolve_device,
    save_predictions_npz,
)
from ship_motion.models import build_model_from_config
from ship_motion.utils import ensure_dir, load_yaml, save_json, set_seed


def build_model(config: dict[str, Any]) -> nn.Module:
    """根据当前配置构造模型。"""
    model_cfg = build_runtime_model_config(config)
    return build_model_from_config(model_cfg)


def format_duration(seconds: float) -> str:
    """把秒数格式化成适合终端阅读的时长字符串。"""
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


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
    non_blocking: bool = False,
    epoch: int | None = None,
    total_epochs: int | None = None,
    show_progress: bool = True,
) -> float:
    """执行一个 epoch 的标准化空间 MSE 训练。"""
    criterion = nn.MSELoss()
    model.train()
    losses: list[float] = []
    total_steps = resolve_progress_total(loader, max_steps=max_steps)
    if epoch is not None and total_epochs is not None:
        progress_desc = f"train[{epoch}/{total_epochs}]"
    else:
        progress_desc = "train"
    use_tqdm = show_progress and progress_is_interactive()
    progress_bar = tqdm(total=total_steps, desc=progress_desc, dynamic_ncols=True, leave=False) if use_tqdm else None

    for step, batch in enumerate(loader, start=1):
        if max_steps is not None and step > max_steps:
            break
        batch = move_batch_to_device(batch, device, non_blocking=non_blocking)
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
        avg_loss = float(sum(losses) / len(losses))
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")
        elif show_progress and (step == 1 or step % 10 == 0 or step == total_steps):
            total_hint = total_steps if total_steps is not None else "?"
            epoch_hint = f"[{epoch}/{total_epochs}] " if epoch is not None and total_epochs is not None else ""
            print(f"[Train] {epoch_hint}step {step}/{total_hint} loss={loss.item():.4f} avg={avg_loss:.4f}")

    if progress_bar is not None:
        progress_bar.close()

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
    non_blocking: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
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
        non_blocking=non_blocking,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )


def fit(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """训练并评估一个 Step 02 基线模型。"""
    runtime_config = copy.deepcopy(config)
    if smoke:
        apply_smoke_overrides(runtime_config)

    train_cfg = runtime_config.get("train", {})
    set_seed(int(train_cfg.get("seed", 42)))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    repo_root = Path(runtime_config["__config_path__"]).resolve().parent.parent
    total_train_start = time.perf_counter()
    log_runtime_environment(device=device, repo_root=repo_root)
    data_prep_start = time.perf_counter()
    bundle = build_datasets(runtime_config, smoke=False)
    data_prep_elapsed = time.perf_counter() - data_prep_start
    data_cfg = runtime_config.get("data", {})
    run_dir = ensure_dir(default_run_dir(runtime_config))

    # 保存“本次真正运行时使用的配置”和 scaler，
    # 后续复现实验或单独评估时会很有用。
    save_runtime_config(runtime_config, run_dir / "config.yaml")
    bundle["scaler"].save(run_dir / "scaler.json")

    loader_settings = resolve_loader_settings(train_cfg, device)
    loaders = build_loaders(
        bundle=bundle,
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=loader_settings["num_workers"],
        pin_memory=loader_settings["pin_memory"],
        persistent_workers=loader_settings["persistent_workers"],
        prefetch_factor=loader_settings["prefetch_factor"],
    )
    model = build_model(runtime_config).to(device)
    optimizer = build_optimizer(model, train_cfg)
    max_eval_steps = train_cfg.get("max_eval_steps")
    lambda_vmd = float(runtime_config.get("vmd", {}).get("lambda_vmd", 0.0))
    physics_cfg = runtime_config.get("physics", {})
    run_name = str(runtime_config.get("run_name", "run"))
    model_name = str(runtime_config.get("model", {}).get("name", "unknown_model"))
    device_label = str(device)
    print("=" * 80)
    print(
        f"[Run Start] model={model_name} | run={run_name} | smoke={smoke} | "
        f"device={device_label} | output={run_dir}"
    )
    print(
        f"[Run Start] data_prep_elapsed={format_duration(data_prep_elapsed)} | "
        f"train_files={len(bundle['train_files'])} | val_files={len(bundle['val_files'])} | "
        f"routine_test_files={len(bundle['routine_test_files'])} | ood_test_files={len(bundle['ood_test_files'])}"
    )
    print(
        f"[Run Start] batch_size={int(train_cfg.get('batch_size', 128))} | "
        f"epochs={int(train_cfg.get('epochs', 1))} | "
        f"train_windows={len(bundle['train'])} | val_windows={len(bundle['val'])}"
    )
    print(
        f"[Run Start] num_workers={loader_settings['num_workers']} | "
        f"pin_memory={loader_settings['pin_memory']} | "
        f"persistent_workers={loader_settings['persistent_workers']} | "
        f"prefetch_factor={loader_settings['prefetch_factor']} | "
        f"non_blocking={loader_settings['non_blocking']}"
    )
    print("=" * 80)

    # Persistence 没有可训练参数，会走这个分支：
    # 不训练，只直接评估并输出统一格式文件。
    if optimizer is None:
        print("[Train] 当前模型无可训练参数，直接进入评估流程。")
        val_metrics, val_loss = validate(
            model=model,
            loader=loaders["val"],
            scaler=bundle["scaler"],
            target_cols=data_cfg["target_cols"],
            device=device,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            max_steps=max_eval_steps,
            non_blocking=loader_settings["non_blocking"],
            show_progress=True,
            progress_desc="eval[val]",
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
        print(
            f"[Run Done] model={model_name} | run={run_name} | "
            f"total_elapsed={format_duration(time.perf_counter() - total_train_start)}"
        )
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
        train_steps = resolve_progress_total(loaders["train"], max_steps=max_train_steps)
        print(
            f"[Train] epoch {epoch}/{epochs} started | model={model_name} | "
            f"run={run_name} | steps={train_steps if train_steps is not None else '?'} | device={device_label}"
        )
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
            non_blocking=loader_settings["non_blocking"],
            epoch=epoch,
            total_epochs=epochs,
            show_progress=True,
        )
        val_steps = resolve_progress_total(loaders["val"], max_steps=max_eval_steps)
        print(f"[Eval] validating epoch {epoch}/{epochs} | steps={val_steps if val_steps is not None else '?'}")
        val_metrics, val_loss = validate(
            model=model,
            loader=loaders["val"],
            scaler=bundle["scaler"],
            target_cols=data_cfg["target_cols"],
            device=device,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            max_steps=max_eval_steps,
            non_blocking=loader_settings["non_blocking"],
            show_progress=True,
            progress_desc="eval[val]",
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
        avg_epoch_time = float(sum(item["elapsed_sec"] for item in history) / len(history))
        remaining_epochs = max(epochs - epoch, 0)
        eta_text = format_duration(avg_epoch_time * remaining_epochs)
        print(
            f"[Epoch Summary] epoch={epoch} train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_rmse_mean={val_metrics['rmse_mean']:.6f} "
            f"elapsed={format_duration(elapsed)} eta={eta_text}"
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
                print(
                    f"[Train] Early stopping triggered at epoch {epoch}. "
                    f"best_epoch={best_epoch} best_val_rmse_mean={best_val_rmse:.6f}"
                )
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
    print("=" * 80)
    print(
        f"[Run Done] model={model_name} | run={run_name} | best_epoch={best_epoch} | "
        f"best_val_rmse_mean={best_val_rmse:.6f} | "
        f"total_elapsed={format_duration(time.perf_counter() - total_train_start)}"
    )
    print("=" * 80)
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
        collect_predictions = split in {"routine_test", "ood_test"}
        print(
            f"[Eval] running split={split} | model={config.get('model', {}).get('name', 'unknown_model')} "
            f"| run={config.get('run_name', 'run')}"
        )
        metrics, _, payload = evaluate_model_with_predictions(
            model=model,
            loader=loaders[split],
            scaler=bundle["scaler"],
            target_cols=data_cfg["target_cols"],
            device=device,
            lambda_vmd=float(config.get("vmd", {}).get("lambda_vmd", 0.0)),
            physics_cfg=config.get("physics", {}),
            max_steps=max_eval_steps,
            non_blocking=resolve_loader_settings(config.get("train", {}), device)["non_blocking"],
            collect_predictions=collect_predictions,
            show_progress=True,
            progress_desc=f"eval[{split}]",
        )
        save_json(metrics, run_dir / f"metrics_{split}.json")
        if collect_predictions and payload is not None:
            save_predictions_npz(payload, run_dir / f"predictions_{split}.npz")
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
    train_cfg["pin_memory"] = False
    train_cfg["persistent_workers"] = False
    train_cfg["prefetch_factor"] = None
    train_cfg["non_blocking"] = False
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

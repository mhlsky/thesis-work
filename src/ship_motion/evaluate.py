from __future__ import annotations

"""Step 02 模型评估工具。"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ship_motion.data.dataset import DEFAULT_CONFIG_PATH, build_datasets
from ship_motion.losses.physics import physics_loss
from ship_motion.losses.vmd_loss import standardize_y_modes, vmd_aux_loss
from ship_motion.metrics import compute_metrics, roll_consistency_rmse, smoothness_metric
from ship_motion.models import build_model_from_config
from ship_motion.utils import ensure_dir, load_yaml, save_json, set_seed


def resolve_progress_total(loader: DataLoader, max_steps: int | None = None) -> int | None:
    """推导进度条总步数；如果无法安全获取长度，则返回 None。"""
    try:
        total_steps = len(loader)
    except TypeError:
        total_steps = None
    if total_steps is None:
        return max_steps
    if max_steps is None:
        return total_steps
    return min(total_steps, max_steps)


def progress_is_interactive() -> bool:
    """判断当前是否适合显示 tqdm 动态进度条。"""
    return bool(getattr(sys.stderr, "isatty", lambda: False)())


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: Any,
    target_cols: Sequence[str],
    device: torch.device,
    lambda_vmd: float = 0.0,
    physics_cfg: dict[str, Any] | None = None,
    max_steps: int | None = None,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[dict[str, float], float]:
    """返回反标准化指标和标准化空间平均 MSE。"""
    metrics, loss_value, _ = evaluate_model_with_predictions(
        model=model,
        loader=loader,
        scaler=scaler,
        target_cols=target_cols,
        device=device,
        lambda_vmd=lambda_vmd,
        physics_cfg=physics_cfg,
        max_steps=max_steps,
        collect_predictions=False,
        show_progress=show_progress,
        progress_desc=progress_desc,
    )
    return metrics, loss_value


def evaluate_model_with_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: Any,
    target_cols: Sequence[str],
    device: torch.device,
    lambda_vmd: float = 0.0,
    physics_cfg: dict[str, Any] | None = None,
    max_steps: int | None = None,
    collect_predictions: bool = True,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[dict[str, float], float, dict[str, np.ndarray] | None]:
    """评估模型，并可选收集预测结果用于后续绘图。"""
    criterion = torch.nn.MSELoss()
    model.eval()
    losses: list[float] = []
    preds: list[torch.Tensor] = []
    trues: list[torch.Tensor] = []
    last_states: list[torch.Tensor] = []
    total_steps = resolve_progress_total(loader, max_steps=max_steps)
    progress_desc = progress_desc or "eval"
    use_tqdm = show_progress and progress_is_interactive()

    with torch.no_grad():
        progress_bar = tqdm(total=total_steps, desc=progress_desc, dynamic_ncols=True, leave=False) if use_tqdm else None
        for step, batch in enumerate(loader, start=1):
            if max_steps is not None and step > max_steps:
                break
            batch = move_batch_to_device(batch, device)
            output = model(batch["x"], batch=batch)
            y_hat, loss, _, _ = compute_model_losses(
                model_output=output,
                batch=batch,
                criterion=criterion,
                scaler=scaler,
                y_std=scaler.y_std,
                lambda_vmd=lambda_vmd,
                physics_cfg=physics_cfg,
            )
            losses.append(float(loss.item()))
            avg_loss = float(sum(losses) / len(losses))
            if progress_bar is not None:
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}")
            elif show_progress and (step == 1 or step % 10 == 0 or step == total_steps):
                total_hint = total_steps if total_steps is not None else "?"
                print(f"[Eval] {progress_desc} step {step}/{total_hint} loss={loss.item():.4f} avg={avg_loss:.4f}")

            # 预测需要先反标准化后再统计最终指标；
            # 真实标签直接使用数据集已经提供好的 y_raw。
            preds.append(scaler.inverse_y_tensor(y_hat).detach().cpu())
            trues.append(batch["y_raw"].detach().cpu())
            last_states.append(batch["last_state_raw"].detach().cpu())
        if progress_bar is not None:
            progress_bar.close()

    if not preds:
        raise ValueError("Evaluation loader produced no batches.")

    y_pred_raw = torch.cat(preds, dim=0)
    y_true_raw = torch.cat(trues, dim=0)
    last_state_raw = torch.cat(last_states, dim=0)
    metrics = compute_metrics(y_pred_raw, y_true_raw, target_cols)
    physics_enabled = bool((physics_cfg or {}).get("enabled", False))
    if physics_enabled:
        dt = float((physics_cfg or {}).get("dt", 1.0))
        metrics["smoothness"] = smoothness_metric(y_pred_raw)
        metrics["roll_consistency_rmse"] = roll_consistency_rmse(y_pred_raw, last_state_raw, dt=dt)
    metrics["num_samples"] = int(y_pred_raw.shape[0])
    metrics["num_forecast_steps"] = int(y_pred_raw.shape[1])
    metrics["loss_mse_std"] = float(sum(losses) / len(losses))
    prediction_payload = None
    if collect_predictions:
        prediction_payload = {
            "y_pred": y_pred_raw.numpy().astype(np.float32),
            "y_true": y_true_raw.numpy().astype(np.float32),
            "last_state_raw": last_state_raw.numpy().astype(np.float32),
        }
    return metrics, float(sum(losses) / len(losses)), prediction_payload


def extract_y_hat(model_output: Any) -> torch.Tensor:
    """统一兼容 Tensor 输出和 dict 输出。"""
    if isinstance(model_output, dict):
        if "y_hat" not in model_output:
            raise KeyError("Model output dict must contain key 'y_hat'.")
        return model_output["y_hat"]
    return model_output


def compute_model_losses(
    model_output: Any,
    batch: dict[str, Any],
    criterion: torch.nn.Module,
    scaler: Any | None = None,
    y_std: Sequence[float] | None = None,
    lambda_vmd: float = 0.0,
    physics_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """统一计算主预测损失和可选的 VMD 辅助损失。"""
    y_hat = extract_y_hat(model_output)
    pred_loss = criterion(y_hat, batch["y"])
    total_loss = pred_loss
    aux_loss: torch.Tensor | None = None

    if (
        isinstance(model_output, dict)
        and "mode_preds" in model_output
        and "y_modes" in batch
        and y_std is not None
        and lambda_vmd > 0
    ):
        y_modes_std = standardize_y_modes(
            batch["y_modes"],
            y_std=y_std,
            reference_tensor=model_output["mode_preds"],
        )
        aux_loss = vmd_aux_loss(model_output["mode_preds"], y_modes_std)
        total_loss = total_loss + lambda_vmd * aux_loss

    physics_enabled = bool((physics_cfg or {}).get("enabled", False))
    if physics_enabled:
        if scaler is None:
            raise ValueError("Physics loss requires scaler for inverse transform.")
        y_hat_raw = scaler.inverse_y_tensor(y_hat)
        physics_total, _, _ = physics_loss(
            y_raw=y_hat_raw,
            last_state_raw=batch["last_state_raw"],
            lambda_smooth=float((physics_cfg or {}).get("lambda_smooth", 0.0)),
            lambda_roll=float((physics_cfg or {}).get("lambda_roll", 0.0)),
            dt=float((physics_cfg or {}).get("dt", 1.0)),
        )
        total_loss = total_loss + physics_total

    return y_hat, total_loss, pred_loss, aux_loss


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """把 batch 中的 Tensor 迁移到指定设备，字符串等元数据保持原样。"""
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def evaluate_splits(
    model: torch.nn.Module,
    loaders: dict[str, DataLoader],
    scaler: Any,
    target_cols: Sequence[str],
    device: torch.device,
    split_names: Iterable[str],
    lambda_vmd: float = 0.0,
    physics_cfg: dict[str, Any] | None = None,
    max_steps: int | None = None,
    show_progress: bool = False,
) -> dict[str, dict[str, float]]:
    """按 split 逐个评估。"""
    results: dict[str, dict[str, float]] = {}
    for split in split_names:
        metrics, _ = evaluate_model(
            model=model,
            loader=loaders[split],
            scaler=scaler,
            target_cols=target_cols,
            device=device,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            max_steps=max_steps,
            show_progress=show_progress,
            progress_desc=f"eval[{split}]",
        )
        results[split] = metrics
    return results


def save_predictions_npz(predictions: dict[str, np.ndarray], path: str | Path) -> None:
    """把预测结果保存成 npz，供 Step 07 绘图与对比使用。"""
    target = Path(path)
    ensure_dir(target.parent)
    np.savez_compressed(target, **predictions)


def build_loaders(bundle: dict[str, Any], batch_size: int, num_workers: int) -> dict[str, DataLoader]:
    """构造 train/val/test DataLoader。"""
    return {
        "train": DataLoader(bundle["train"], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(bundle["val"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
        "routine_test": DataLoader(
            bundle["routine_test"], batch_size=batch_size, shuffle=False, num_workers=num_workers
        ),
        "ood_test": DataLoader(bundle["ood_test"], batch_size=batch_size, shuffle=False, num_workers=num_workers),
    }


def load_model_from_config(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    device: torch.device,
) -> torch.nn.Module:
    """根据配置创建模型，并在提供 checkpoint 时加载权重。"""
    model_cfg = build_runtime_model_config(config)
    model = build_model_from_config(model_cfg).to(device)
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
    return model


def build_runtime_model_config(config: dict[str, Any]) -> dict[str, Any]:
    """把 data 与 model 配置合并成真正建模时使用的参数字典。"""
    model_cfg = copy.deepcopy(config.get("model", {}))
    data_cfg = config.get("data", {})
    model_cfg.setdefault("input_dim", len(data_cfg.get("input_cols", [])))
    model_cfg.setdefault("target_dim", len(data_cfg.get("target_cols", [])))
    model_cfg.setdefault("exog_dim", len(data_cfg.get("exog_cols", [])))
    model_cfg.setdefault("state_dim", len(data_cfg.get("state_cols", [])))
    if "vmd" in config and "K" in config.get("vmd", {}):
        model_cfg.setdefault("K", int(config["vmd"]["K"]))
    model_cfg["pred_len"] = int(data_cfg["pred_len"])
    return model_cfg


def resolve_device(device_name: str) -> torch.device:
    """把 auto/cpu/cuda 这样的字符串解析成 torch.device。"""
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def default_run_dir(config: dict[str, Any]) -> Path:
    """根据配置推导默认输出目录。"""
    config_path = Path(config["__config_path__"])
    repo_root = config_path.parent.parent
    output_root = Path(config.get("output_dir", "outputs"))
    if not output_root.is_absolute():
        output_root = (repo_root / output_root).resolve()
    return output_root / config["run_name"]


def evaluate_from_config(
    config_path: str | Path,
    checkpoint: str | Path | None = None,
    split: str = "all",
    smoke: bool = False,
    save: bool = True,
    save_predictions: bool = False,
) -> dict[str, dict[str, float]]:
    """独立评估命令入口。"""
    config = load_yaml(config_path)
    if smoke:
        apply_smoke_overrides(config)

    set_seed(int(config.get("train", {}).get("seed", 42)))
    bundle = build_datasets(config, smoke=False)
    train_cfg = config.get("train", {})
    device = resolve_device(str(train_cfg.get("device", "auto")))
    loaders = build_loaders(
        bundle=bundle,
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=int(train_cfg.get("num_workers", 0)),
    )

    split_names = [split] if split != "all" else ["val", "routine_test", "ood_test"]
    model = load_model_from_config(config, checkpoint_path=checkpoint, device=device)
    lambda_vmd = float(config.get("vmd", {}).get("lambda_vmd", 0.0))
    physics_cfg = config.get("physics", {})
    results: dict[str, dict[str, float]] = {}
    prediction_payloads: dict[str, dict[str, np.ndarray]] = {}
    for split_name in split_names:
        metrics, _, payload = evaluate_model_with_predictions(
            model=model,
            loader=loaders[split_name],
            scaler=bundle["scaler"],
            target_cols=config["data"]["target_cols"],
            device=device,
            lambda_vmd=lambda_vmd,
            physics_cfg=physics_cfg,
            max_steps=train_cfg.get("max_eval_steps"),
            collect_predictions=save_predictions,
            show_progress=True,
            progress_desc=f"eval[{split_name}]",
        )
        results[split_name] = metrics
        if save_predictions and payload is not None:
            prediction_payloads[split_name] = payload

    if save:
        run_dir = ensure_dir(default_run_dir(config))
        for split_name, metrics in results.items():
            save_json(metrics, run_dir / f"metrics_{split_name}.json")
            if save_predictions and split_name in prediction_payloads:
                save_predictions_npz(prediction_payloads[split_name], run_dir / f"predictions_{split_name}.npz")
    return results


def apply_smoke_overrides(config: dict[str, Any]) -> None:
    """给评估流程套上更小的数据量和输出目录。"""
    config["run_name"] = f"{config.get('run_name', 'run')}_smoke"
    config["output_dir"] = "outputs/smoke"
    data_cfg = config.setdefault("data", {})
    data_cfg["max_files"] = min(int(data_cfg.get("max_files") or 1), 1)
    data_cfg["max_windows_per_file"] = min(int(data_cfg.get("max_windows_per_file") or 8), 8)
    train_cfg = config.setdefault("train", {})
    train_cfg["batch_size"] = min(int(train_cfg.get("batch_size", 8)), 8)
    train_cfg["num_workers"] = 0
    train_cfg["max_eval_steps"] = 2
    apply_vmd_smoke_overrides(config)


def apply_vmd_smoke_overrides(config: dict[str, Any]) -> None:
    """把 VMD 缓存路径指向 smoke 目录，避免要求正式全量缓存。"""
    vmd_cfg = config.get("vmd")
    if not isinstance(vmd_cfg, dict) or not bool(vmd_cfg.get("enabled", False)):
        return
    alpha = float(vmd_cfg.get("alpha", 2000.0))
    alpha_str = f"{alpha:g}".replace(".", "_")
    cache_dir_name = f"K{int(vmd_cfg.get('K', 3))}_alpha{alpha_str}"
    vmd_cfg["cache_root"] = f"outputs/cache/vmd_smoke/{cache_dir_name}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a baseline ship-motion model.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", choices=["all", "val", "routine_test", "ood_test"], default="all")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args(argv)

    results = evaluate_from_config(
        config_path=args.config,
        checkpoint=args.checkpoint,
        split=args.split,
        smoke=args.smoke,
        save=not args.no_save,
        save_predictions=args.save_predictions,
    )
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

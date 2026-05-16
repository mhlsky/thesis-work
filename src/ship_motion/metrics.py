from __future__ import annotations

"""评估指标。

这里的核心规则非常重要：
所有指标都假设输入已经处在真实物理尺度上，
因此训练/评估流程必须先把预测和标签从标准化空间反变换回来。

换句话说：
- 训练 loss 在标准化空间里算；
- 最终 MAE / RMSE / R² 在真实物理量空间里算。
"""

from typing import Any, Sequence

import numpy as np


def mae(y_pred: Any, y_true: Any) -> float:
    """平均绝对误差。"""
    pred, true = _to_numpy_pair(y_pred, y_true)
    return float(np.mean(np.abs(pred - true)))


def rmse(y_pred: Any, y_true: Any) -> float:
    """均方根误差。"""
    pred, true = _to_numpy_pair(y_pred, y_true)
    return float(np.sqrt(np.mean(np.square(pred - true))))


def r2_score(y_pred: Any, y_true: Any) -> float:
    """决定系数 R²。"""
    pred, true = _to_numpy_pair(y_pred, y_true)
    true_mean = np.mean(true)
    ss_res = float(np.sum(np.square(true - pred)))
    ss_tot = float(np.sum(np.square(true - true_mean)))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(y_pred_raw: Any, y_true_raw: Any, target_cols: Sequence[str]) -> dict[str, float]:
    """同时输出整体均值指标和每个目标维度的细分指标。"""
    pred, true = _to_numpy_pair(y_pred_raw, y_true_raw)
    if pred.shape != true.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, true={true.shape}")
    if pred.shape[-1] != len(target_cols):
        raise ValueError(
            f"Last dimension {pred.shape[-1]} does not match target_cols size {len(target_cols)}"
        )

    # 把 [B, pred_len, C] 展平成 [B * pred_len, C]，
    # 这样可以统一对每个目标维度统计总体误差。
    pred_flat = pred.reshape(-1, pred.shape[-1])
    true_flat = true.reshape(-1, true.shape[-1])

    metrics: dict[str, float] = {
        "mae_mean": mae(pred_flat, true_flat),
        "rmse_mean": rmse(pred_flat, true_flat),
        "r2_mean": r2_score(pred_flat, true_flat),
    }

    maes: list[float] = []
    rmses: list[float] = []
    r2s: list[float] = []
    for index, name in enumerate(target_cols):
        pred_col = pred_flat[:, index]
        true_col = true_flat[:, index]
        mae_value = mae(pred_col, true_col)
        rmse_value = rmse(pred_col, true_col)
        r2_value = r2_score(pred_col, true_col)
        maes.append(mae_value)
        rmses.append(rmse_value)
        r2s.append(r2_value)
        metrics[f"mae_{name}"] = mae_value
        metrics[f"rmse_{name}"] = rmse_value
        metrics[f"r2_{name}"] = r2_value

    metrics["mae_mean_per_target"] = float(np.mean(maes))
    metrics["rmse_mean_per_target"] = float(np.mean(rmses))
    metrics["r2_mean_per_target"] = float(np.mean(r2s))
    return metrics


def _to_numpy_pair(y_pred: Any, y_true: Any) -> tuple[np.ndarray, np.ndarray]:
    pred = _to_numpy(y_pred)
    true = _to_numpy(y_true)
    return pred.astype(np.float64), true.astype(np.float64)


def _to_numpy(value: Any) -> np.ndarray:
    """兼容 torch.Tensor、numpy 数组和普通列表。"""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)

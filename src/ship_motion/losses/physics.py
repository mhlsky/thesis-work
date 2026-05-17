from __future__ import annotations

"""Step 06：物理约束损失。"""

from typing import Any


def smoothness_loss(y_raw: Any) -> Any:
    """二阶差分平滑约束。

    y_raw: [B, pred_len, target_dim]
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for smoothness_loss. Please run `uv sync` first.") from exc

    if y_raw.size(1) < 3:
        return torch.zeros((), device=y_raw.device, dtype=y_raw.dtype)
    d1 = y_raw[:, 1:, :] - y_raw[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return torch.mean(d2.square())


def roll_kinematic_loss(
    y_raw: Any,
    last_state_raw: Any,
    dt: float = 1.0,
    p_idx: int = 2,
    phi_idx: int = 4,
) -> Any:
    """横摇角速度 p 与横摇角 phi 的近似积分一致性约束。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for roll_kinematic_loss. Please run `uv sync` first."
        ) from exc

    phi_hat = y_raw[:, :, phi_idx]
    p_hat = y_raw[:, :, p_idx]
    phi_last = last_state_raw[:, phi_idx]
    p_last = last_state_raw[:, p_idx]

    # 第一个预测步需要和历史最后一帧衔接。
    first_residual = phi_hat[:, 0] - phi_last - p_last * dt
    residuals = [first_residual]

    if y_raw.size(1) > 1:
        inner_residual = phi_hat[:, 1:] - phi_hat[:, :-1] - p_hat[:, :-1] * dt
        residuals.append(inner_residual.reshape(-1))

    stacked = torch.cat([item.reshape(-1) for item in residuals], dim=0)
    return torch.mean(stacked.square())


def physics_loss(
    y_raw: Any,
    last_state_raw: Any,
    lambda_smooth: float,
    lambda_roll: float,
    dt: float = 1.0,
    p_idx: int = 2,
    phi_idx: int = 4,
) -> tuple[Any, Any, Any]:
    """返回总物理损失、平滑损失、横摇运动学损失。"""
    smooth = smoothness_loss(y_raw)
    roll = roll_kinematic_loss(
        y_raw=y_raw,
        last_state_raw=last_state_raw,
        dt=dt,
        p_idx=p_idx,
        phi_idx=phi_idx,
    )
    total = lambda_smooth * smooth + lambda_roll * roll
    return total, smooth, roll

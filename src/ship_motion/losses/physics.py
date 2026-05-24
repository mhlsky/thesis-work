from __future__ import annotations

"""Step 06：物理约束损失。"""

from typing import Any, Sequence


def smoothness_loss(
    y_raw: Any,
    target_indices: Sequence[int] | None = None,
    target_scales: Sequence[float] | Any | None = None,
) -> Any:
    """二阶差分平滑约束。

    y_raw: [B, pred_len, target_dim]
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for smoothness_loss. Please run `uv sync` first.") from exc

    if y_raw.size(1) < 3:
        return torch.zeros((), device=y_raw.device, dtype=y_raw.dtype)

    y_selected = _select_target_dims(y_raw, target_indices)
    if y_selected.size(-1) == 0:
        return torch.zeros((), device=y_raw.device, dtype=y_raw.dtype)

    if target_scales is not None:
        scale_tensor = _build_scale_tensor(
            reference_tensor=y_selected,
            target_indices=target_indices,
            target_scales=target_scales,
        )
        y_selected = y_selected / scale_tensor

    d1 = y_selected[:, 1:, :] - y_selected[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return torch.mean(d2.square())


def roll_kinematic_loss(
    y_raw: Any,
    last_state_raw: Any,
    dt: float = 1.0,
    p_idx: int = 2,
    phi_idx: int = 4,
    integration: str = "euler",
    residual_scale: float | None = None,
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

    integration_name = str(integration).strip().lower()
    if integration_name not in {"euler", "trapezoid"}:
        raise ValueError(f"Unsupported roll integration method: {integration!r}")

    # 第一个预测步需要和历史最后一帧衔接。
    if integration_name == "trapezoid":
        first_residual = phi_hat[:, 0] - phi_last - 0.5 * (p_last + p_hat[:, 0]) * dt
    else:
        first_residual = phi_hat[:, 0] - phi_last - p_last * dt
    residuals = [first_residual]

    if y_raw.size(1) > 1:
        if integration_name == "trapezoid":
            inner_residual = phi_hat[:, 1:] - phi_hat[:, :-1] - 0.5 * (p_hat[:, 1:] + p_hat[:, :-1]) * dt
        else:
            inner_residual = phi_hat[:, 1:] - phi_hat[:, :-1] - p_hat[:, :-1] * dt
        residuals.append(inner_residual.reshape(-1))

    stacked = torch.cat([item.reshape(-1) for item in residuals], dim=0)
    if residual_scale is not None and residual_scale > 0:
        stacked = stacked / float(residual_scale)
    return torch.mean(stacked.square())


def physics_loss(
    y_raw: Any,
    last_state_raw: Any,
    lambda_smooth: float,
    lambda_roll: float,
    dt: float = 1.0,
    p_idx: int = 2,
    phi_idx: int = 4,
    smoothness_target_indices: Sequence[int] | None = None,
    target_scales: Sequence[float] | Any | None = None,
    roll_integration: str = "euler",
    roll_residual_scale: float | None = None,
) -> tuple[Any, Any, Any]:
    """返回总物理损失、平滑损失、横摇运动学损失。"""
    smooth = smoothness_loss(
        y_raw=y_raw,
        target_indices=smoothness_target_indices,
        target_scales=target_scales,
    )
    roll = roll_kinematic_loss(
        y_raw=y_raw,
        last_state_raw=last_state_raw,
        dt=dt,
        p_idx=p_idx,
        phi_idx=phi_idx,
        integration=roll_integration,
        residual_scale=roll_residual_scale,
    )
    total = lambda_smooth * smooth + lambda_roll * roll
    return total, smooth, roll


def _select_target_dims(y_raw: Any, target_indices: Sequence[int] | None) -> Any:
    """按目标维索引筛选需要施加平滑约束的变量。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for _select_target_dims.") from exc

    if not target_indices:
        return y_raw
    index_tensor = torch.as_tensor(list(target_indices), device=y_raw.device, dtype=torch.long)
    return torch.index_select(y_raw, dim=-1, index=index_tensor)


def _build_scale_tensor(
    reference_tensor: Any,
    target_indices: Sequence[int] | None,
    target_scales: Sequence[float] | Any,
) -> Any:
    """构造可广播的尺度张量，用于按目标变量标准差归一化物理损失。"""
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("PyTorch is required for _build_scale_tensor.") from exc

    if isinstance(target_scales, torch.Tensor):
        scale_values = target_scales.to(device=reference_tensor.device, dtype=reference_tensor.dtype)
    else:
        scale_values = reference_tensor.new_tensor(list(target_scales))
    if target_indices:
        index_tensor = torch.as_tensor(list(target_indices), device=reference_tensor.device, dtype=torch.long)
        scale_values = torch.index_select(scale_values, dim=0, index=index_tensor)
    scale_values = torch.clamp(scale_values, min=1.0e-8)
    shape = [1] * (reference_tensor.dim() - 1) + [int(scale_values.numel())]
    return scale_values.view(*shape)

from __future__ import annotations

"""VMD 辅助监督损失。"""

from typing import Any, Sequence


def standardize_y_modes(y_modes: Any, y_std: Sequence[float] | Any, reference_tensor: Any | None = None) -> Any:
    """把原始物理尺度的 VMD 模态缩放到标准化空间。

    规则：
    y_modes_std = y_modes_raw / y_std

    注意这里不减 y_mean。
    原因是多个模态相加后才对应完整原始信号，
    如果每个模态都重复减均值，会导致目标定义失真。
    """
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for standardize_y_modes. Please run `uv sync` first."
        ) from exc

    if isinstance(y_std, torch.Tensor):
        std_tensor = y_std
    else:
        base_tensor = reference_tensor if reference_tensor is not None else y_modes
        std_tensor = base_tensor.new_tensor(list(y_std))

    shape = [1] * (y_modes.dim() - 2) + [len(std_tensor), 1]
    std_tensor = std_tensor.view(*shape)
    return y_modes / std_tensor


def vmd_aux_loss(mode_preds: Any, y_modes: Any) -> Any:
    """计算 VMD 模态预测与真实模态之间的 MSE。

    参数约定：
    - mode_preds: [B, pred_len, target_dim, K]
    - y_modes:    [B, pred_len, target_dim, K]

    本函数目前只提供最基础的统一入口，后续 Step 05
    可直接在训练循环里复用，不需要再重复写维度说明。
    """
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for vmd_aux_loss. Please run `uv sync` first."
        ) from exc

    criterion = nn.MSELoss()
    return criterion(mode_preds, y_modes)

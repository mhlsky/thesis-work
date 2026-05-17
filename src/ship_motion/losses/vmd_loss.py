from __future__ import annotations

"""VMD 辅助监督损失。"""

from typing import Any


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

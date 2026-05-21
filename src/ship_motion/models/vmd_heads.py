from __future__ import annotations

"""Step 05：VMD 多分支预测头。"""

import torch
from torch import nn


class VMDMultiHead(nn.Module):
    """根据 backbone 特征一次性预测全部 VMD 模态。

    为了保持实现简单，这里不拆成 K 个独立 head，
    而是直接用一个 MLP 输出 `[pred_len, target_dim, K]`。
    """

    def __init__(self, d_model: int, pred_len: int, target_dim: int, K: int, dropout: float = 0.1) -> None:
        super().__init__()
        if K <= 0:
            raise ValueError("K must be positive for VMDMultiHead.")
        self.pred_len = pred_len
        self.target_dim = target_dim
        self.K = K
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * target_dim * K),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch_size = feature.size(0)
        output = self.net(feature)
        return output.view(batch_size, self.pred_len, self.target_dim, self.K)

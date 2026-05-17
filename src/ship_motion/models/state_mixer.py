from __future__ import annotations

"""Step 05：状态耦合修正模块。"""

import torch
from torch import nn


class StateCouplingMixer(nn.Module):
    """在每个预测步内部，对 5 个状态量做一次轻量残差修正。"""

    def __init__(self, target_dim: int = 5, hidden_dim: int = 16, dropout: float = 0.0) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.net = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        batch_size, pred_len, target_dim = y.shape
        if target_dim != self.target_dim:
            raise ValueError(f"StateCouplingMixer expected target_dim={self.target_dim}, got {target_dim}.")
        residual = self.net(y.reshape(batch_size * pred_len, target_dim)).view(batch_size, pred_len, target_dim)
        return y + residual

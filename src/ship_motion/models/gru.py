from __future__ import annotations

import torch
from torch import nn


class GRUForecaster(nn.Module):
    """标准 GRU 多步预测基线。

    GRU 和 LSTM 的整体框架非常接近，
    但门控结构更轻，参数通常也更少，
    适合作为轻量 RNN 基线。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        pred_len: int,
        target_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * target_dim),
        )
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        """读取最后一个 GRU 隐状态，并输出未来预测窗口。"""
        del batch  # 统一接口保留 batch 参数；当前 GRU 不直接使用它。
        _, hidden = self.encoder(x)
        last_hidden = hidden[-1]
        output = self.head(last_hidden)
        return output.view(x.size(0), self.pred_len, self.target_dim)

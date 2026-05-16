from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """标准正弦位置编码。"""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer 表示：
        # 这个张量会跟着模型一起迁移设备、保存/加载，
        # 但它不是需要训练的参数。
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerForecaster(nn.Module):
    """标准 Transformer Encoder 多步预测基线。"""

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        pred_len: int,
        target_dim: int,
        dropout: float = 0.1,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        dim_feedforward = dim_feedforward or d_model * 4
        self.input_proj = nn.Linear(input_dim, d_model)
        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * target_dim),
        )
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        """用历史窗口编码结果的最后一个 token 生成未来预测。"""
        del batch  # 统一接口保留 batch 参数；当前 Transformer 不直接使用它。

        hidden = self.input_proj(x)
        hidden = self.position(hidden)
        hidden = self.encoder(hidden)
        hidden = self.norm(hidden)

        # 当前采用 last token pooling。
        # 后续如果需要，也可以改成 mean pooling 做对照实验。
        pooled = hidden[:, -1, :]
        output = self.head(pooled)
        return output.view(x.size(0), self.pred_len, self.target_dim)

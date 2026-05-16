from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    """标准 LSTM 多步预测基线。

    结构可以概括成：
    输入 x
      -> LSTM 编码历史序列
      -> 取最后一个隐藏状态
      -> MLP 预测头
      -> reshape 成 [B, pred_len, target_dim]
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

        # PyTorch 的 LSTM 只有在层数大于 1 时才会真正使用内部 dropout。
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        # 预测头先做一次非线性变换，再映射到未来多步目标。
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len * target_dim),
        )
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        """根据历史窗口输出未来多步预测。"""
        del batch  # 统一接口保留 batch 参数；当前 LSTM 不直接使用它。
        _, (hidden, _) = self.encoder(x)

        # hidden 形状是 [num_layers, B, hidden_dim]；
        # 这里取最后一层的隐藏状态作为整个历史序列的摘要。
        last_hidden = hidden[-1]
        output = self.head(last_hidden)
        return output.view(x.size(0), self.pred_len, self.target_dim)

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """裁掉右侧 padding，保证卷积不看到未来信息。

    因果卷积常见实现方式之一是：
    - 先在时间轴右侧补 padding；
    - 再把多补出来的未来部分裁掉。
    """

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size]


class TemporalBlock(nn.Module):
    """TCN 的残差卷积块。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 当通道数变化时，用 1x1 卷积把残差分支投影到同一维度。
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(self.net(x) + residual)


class TCNForecaster(nn.Module):
    """基于因果/膨胀卷积的时间卷积网络基线。"""

    def __init__(
        self,
        input_dim: int,
        channels: list[int],
        kernel_size: int,
        pred_len: int,
        target_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if not channels:
            raise ValueError("channels must not be empty.")

        blocks = []
        in_channels = input_dim
        for index, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**index,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.network = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.Linear(channels[-1], channels[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], pred_len * target_dim),
        )
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        """先做时间卷积，再从最后时刻特征生成未来窗口。"""
        del batch  # 统一接口保留 batch 参数；当前 TCN 不直接使用它。

        # Conv1d 期望输入是 [B, C, L]，所以需要先转置。
        features = self.network(x.transpose(1, 2)).transpose(1, 2)

        # 取最后一个时间步的卷积特征作为历史序列摘要。
        last_feature = features[:, -1, :]
        output = self.head(last_feature)
        return output.view(x.size(0), self.pred_len, self.target_dim)

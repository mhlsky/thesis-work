from __future__ import annotations

"""Step 05：预测解码器。"""

import torch
from torch import nn


class DeltaDecoder(nn.Module):
    """支持 direct / delta 两种预测模式。

    - direct：把输入当作未来状态的直接预测；
    - delta：把输入当作未来增量，并从历史最后状态开始累积。
    """

    def __init__(self, mode: str = "delta") -> None:
        super().__init__()
        normalized_mode = str(mode).lower()
        if normalized_mode not in {"direct", "delta"}:
            raise ValueError(f"Unsupported decode mode: {mode!r}")
        self.mode = normalized_mode

    def forward(self, y_base: torch.Tensor, last_state_std: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "direct":
            return y_base

        if last_state_std is None:
            raise ValueError("DeltaDecoder in delta mode requires last_state_std.")

        # 先累计各步增量，再在每个预测步上加上历史最后一个真实状态。
        cumulative_delta = torch.cumsum(y_base, dim=1)
        return cumulative_delta + last_state_std.unsqueeze(1)

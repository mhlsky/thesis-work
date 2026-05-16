from __future__ import annotations

import torch
from torch import nn


class PersistenceForecaster(nn.Module):
    """Persistence 基线：未来所有步都复制历史最后一个状态。

    这个模型没有可学习参数，因此它的作用主要是：
    - 给出“什么都不学，只复制最后值”的最低基线；
    - 帮我们判断更复杂模型是否真的学到了有效时序规律。
    """

    def __init__(self, pred_len: int, target_dim: int) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        """返回形状为 [B, pred_len, target_dim] 的标准化预测。

        这里优先使用 `batch["x_state"]`，因为它语义最明确：
        - x_state 只包含状态量，不会混入外生控制/环境量；
        - 它与 target_dim 一一对应。

        如果上游 batch 没有显式提供 x_state，就退回到：
        `x[:, -1, -target_dim:]`
        即假设输入最后若干列就是状态列。
        """
        if batch is not None and "x_state" in batch:
            last_state = batch["x_state"][:, -1, : self.target_dim]
        else:
            last_state = x[:, -1, -self.target_dim :]
        return last_state.unsqueeze(1).repeat(1, self.pred_len, 1)

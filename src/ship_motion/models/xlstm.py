from __future__ import annotations

"""Step 04：Lite-xLSTM 与 CCG-xLSTM 主干模型。

本文件实现两个版本：
1. Lite-xLSTM：使用指数门控 + 归一化记忆更新；
2. CCG-xLSTM：在 Lite-xLSTM 的基础上，让外生控制/环境量参与门控。

这里不追求“完整复现官方大模型版 xLSTM”，而是实现一个：
- 更适合当前船舶时序任务；
- 更容易调试；
- 更方便后续接 VMD / 物理约束 / 注意力模块
的轻量 backbone。
"""

import torch
from torch import nn

from .decoders import DeltaDecoder
from .state_mixer import StateCouplingMixer
from .vmd_heads import VMDMultiHead


class ForecastHead(nn.Module):
    """把时序编码结果映射成未来多步状态预测。"""

    def __init__(self, d_model: int, pred_len: int, target_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * target_dim),
        )
        self.pred_len = pred_len
        self.target_dim = target_dim

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch_size = feature.size(0)
        output = self.net(feature)
        return output.view(batch_size, self.pred_len, self.target_dim)


class LiteXLSTMCell(nn.Module):
    """Lite-xLSTM 单元。

    对每个时间步执行：
    i_t = exp(...)
    f_t = exp(...)
    z_t = tanh(...)
    o_t = sigmoid(...)
    c_t = (f_t * c_prev + i_t * z_t) / (f_t + i_t + eps)
    h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int, gate_clip: float = 5.0, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_clip = float(gate_clip)
        self.eps = eps

        # 一次线性层同时生成 4 组门控/候选量，减少重复计算。
        self.gate_proj = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)

    def zero_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回初始隐状态和记忆状态。"""
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return h0, c0

    def forward(
        self,
        x_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """处理单个时间步。

        参数：
        - x_t: 当前时刻输入，形状 [B, input_dim]
        - state: (h_prev, c_prev)
        """
        if state is None:
            h_prev, c_prev = self.zero_state(x_t.size(0), x_t.device, x_t.dtype)
        else:
            h_prev, c_prev = state

        gate_input = torch.cat([x_t, h_prev], dim=-1)
        i_pre, f_pre, z_pre, o_pre = self.gate_proj(gate_input).chunk(4, dim=-1)

        # exp 门控增长很快，所以先做 clip，避免数值过大。
        i_t = torch.exp(torch.clamp(i_pre, min=-self.gate_clip, max=self.gate_clip))
        f_t = torch.exp(torch.clamp(f_pre, min=-self.gate_clip, max=self.gate_clip))
        z_t = torch.tanh(z_pre)
        o_t = torch.sigmoid(o_pre)

        c_t = (f_t * c_prev + i_t * z_t) / (f_t + i_t + self.eps)
        h_t = o_t * torch.tanh(c_t)
        return h_t, (h_t, c_t)


class CCGLiteXLSTMCell(nn.Module):
    """加入控制/环境条件门控的 Lite-xLSTM 单元。

    这里把外生量 exog_t 先编码成 context_t，再加到门控预激活上：
    gate_pre = W[x_t, h_prev] + U context_t
    """

    def __init__(
        self,
        input_dim: int,
        exog_dim: int,
        hidden_dim: int,
        context_dim: int = 64,
        gate_clip: float = 5.0,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gate_clip = float(gate_clip)
        self.eps = eps

        self.gate_proj = nn.Linear(input_dim + hidden_dim, hidden_dim * 4)
        self.context_encoder = nn.Sequential(
            nn.Linear(exog_dim, context_dim),
            nn.GELU(),
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
        )
        self.context_to_gates = nn.Linear(context_dim, hidden_dim * 4)

    def zero_state(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        c0 = torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
        return h0, c0

    def forward(
        self,
        x_t: torch.Tensor,
        x_exog_t: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """处理单个时间步，并让外生量参与门控。"""
        if state is None:
            h_prev, c_prev = self.zero_state(x_t.size(0), x_t.device, x_t.dtype)
        else:
            h_prev, c_prev = state

        gate_input = torch.cat([x_t, h_prev], dim=-1)
        gate_base = self.gate_proj(gate_input)
        context_t = self.context_encoder(x_exog_t)
        gate_context = self.context_to_gates(context_t)
        i_pre, f_pre, z_pre, o_pre = (gate_base + gate_context).chunk(4, dim=-1)

        i_t = torch.exp(torch.clamp(i_pre, min=-self.gate_clip, max=self.gate_clip))
        f_t = torch.exp(torch.clamp(f_pre, min=-self.gate_clip, max=self.gate_clip))
        z_t = torch.tanh(z_pre)
        o_t = torch.sigmoid(o_pre)

        c_t = (f_t * c_prev + i_t * z_t) / (f_t + i_t + self.eps)
        h_t = o_t * torch.tanh(c_t)
        return h_t, (h_t, c_t)


class LiteXLSTMBlock(nn.Module):
    """一个 Lite-xLSTM block：时序递推 + Dropout + Residual + LayerNorm。"""

    def __init__(self, d_model: int, gate_clip: float = 5.0, dropout: float = 0.1) -> None:
        super().__init__()
        self.cell = LiteXLSTMCell(input_dim=d_model, hidden_dim=d_model, gate_clip=gate_clip)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        outputs: list[torch.Tensor] = []
        for step in range(x.size(1)):
            h_t, state = self.cell(x[:, step, :], state)
            outputs.append(h_t)
        hidden_seq = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(hidden_seq))


class CCGLiteXLSTMBlock(nn.Module):
    """一个 CCG-xLSTM block：门控额外读取 x_exog。"""

    def __init__(
        self,
        d_model: int,
        exog_dim: int,
        context_dim: int = 64,
        gate_clip: float = 5.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cell = CCGLiteXLSTMCell(
            input_dim=d_model,
            exog_dim=exog_dim,
            hidden_dim=d_model,
            context_dim=context_dim,
            gate_clip=gate_clip,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, x_exog: torch.Tensor) -> torch.Tensor:
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        outputs: list[torch.Tensor] = []
        for step in range(x.size(1)):
            h_t, state = self.cell(x[:, step, :], x_exog[:, step, :], state)
            outputs.append(h_t)
        hidden_seq = torch.stack(outputs, dim=1)
        return self.norm(x + self.dropout(hidden_seq))


class LiteXLSTMForecaster(nn.Module):
    """Lite-xLSTM 预测器。

    流程：
    x -> Linear input_proj -> 多层 Lite-xLSTM block -> 取最后时间步特征 -> 预测头
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_layers: int,
        pred_len: int,
        target_dim: int,
        dropout: float = 0.1,
        gate_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList(
            [LiteXLSTMBlock(d_model=d_model, gate_clip=gate_clip, dropout=dropout) for _ in range(num_layers)]
        )
        self.head = ForecastHead(d_model=d_model, pred_len=pred_len, target_dim=target_dim, dropout=dropout)

    def encode(
        self,
        x: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回整个隐藏序列和最后时间步特征，方便后续模块复用。"""
        del batch  # Lite 版当前不需要额外 batch 字段，但保留统一接口。
        hidden_seq = self.input_proj(x)
        for layer in self.layers:
            hidden_seq = layer(hidden_seq)
        feature = hidden_seq[:, -1, :]
        return hidden_seq, feature

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        _, feature = self.encode(x, batch=batch)
        return self.head(feature)


class CCGXLSTMForecaster(nn.Module):
    """控制/环境条件门控 xLSTM 预测器。"""

    def __init__(
        self,
        state_dim: int,
        exog_dim: int,
        d_model: int,
        num_layers: int,
        pred_len: int,
        target_dim: int,
        context_dim: int = 64,
        dropout: float = 0.1,
        gate_clip: float = 5.0,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.exog_dim = exog_dim

        # 先把状态量投影到 d_model，便于后续做 residual stack。
        self.state_proj = nn.Linear(state_dim, d_model)
        self.layers = nn.ModuleList(
            [
                CCGLiteXLSTMBlock(
                    d_model=d_model,
                    exog_dim=exog_dim,
                    context_dim=context_dim,
                    gate_clip=gate_clip,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.head = ForecastHead(d_model=d_model, pred_len=pred_len, target_dim=target_dim, dropout=dropout)

    def _split_inputs(self, x: torch.Tensor, batch: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
        """优先使用 Dataset 明确返回的 x_state/x_exog；否则回退到按列切片。"""
        if batch is not None and "x_state" in batch and "x_exog" in batch:
            x_state = batch["x_state"]
            x_exog = batch["x_exog"]
            return x_state, x_exog

        expected_dim = self.exog_dim + self.state_dim
        if x.size(-1) < expected_dim:
            raise ValueError(
                f"CCGXLSTMForecaster expected at least {expected_dim} input features, got {x.size(-1)}."
            )
        x_exog = x[:, :, : self.exog_dim]
        x_state = x[:, :, self.exog_dim : self.exog_dim + self.state_dim]
        return x_state, x_exog

    def encode(
        self,
        x: torch.Tensor,
        batch: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """编码历史序列，并保留隐藏序列供后续增强模块使用。"""
        x_state, x_exog = self._split_inputs(x, batch)
        hidden_seq = self.state_proj(x_state)
        for layer in self.layers:
            hidden_seq = layer(hidden_seq, x_exog)
        feature = hidden_seq[:, -1, :]
        return hidden_seq, feature

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> torch.Tensor:
        _, feature = self.encode(x, batch=batch)
        return self.head(feature)


class VMDCCGXLSTMForecaster(nn.Module):
    """Step 05：VMD-CCG-xLSTM 集成模型。

    流程：
    CCG backbone encode
      -> VMD 多模态预测头
      -> 按模态求和得到基础预测
      -> Delta Decoder（可选 direct / delta）
      -> State Coupling Mixer（可选）
    """

    def __init__(
        self,
        state_dim: int,
        exog_dim: int,
        d_model: int,
        num_layers: int,
        pred_len: int,
        target_dim: int,
        K: int,
        context_dim: int = 64,
        dropout: float = 0.1,
        gate_clip: float = 5.0,
        decode_type: str = "delta",
        use_state_mixer: bool = True,
        state_mixer_hidden_dim: int = 16,
        state_mixer_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.target_dim = target_dim
        self.backbone = CCGXLSTMForecaster(
            state_dim=state_dim,
            exog_dim=exog_dim,
            d_model=d_model,
            num_layers=num_layers,
            pred_len=pred_len,
            target_dim=target_dim,
            context_dim=context_dim,
            dropout=dropout,
            gate_clip=gate_clip,
        )
        self.vmd_head = VMDMultiHead(
            d_model=d_model,
            pred_len=pred_len,
            target_dim=target_dim,
            K=K,
            dropout=dropout,
        )
        self.decoder = DeltaDecoder(mode=decode_type)
        self.use_state_mixer = bool(use_state_mixer)
        self.state_mixer = (
            StateCouplingMixer(
                target_dim=target_dim,
                hidden_dim=state_mixer_hidden_dim,
                dropout=state_mixer_dropout,
            )
            if self.use_state_mixer
            else nn.Identity()
        )

    def _resolve_last_state_std(self, x: torch.Tensor, batch: dict | None) -> torch.Tensor:
        """获取标准化空间中的历史最后状态，供 delta 解码使用。"""
        if batch is not None and "x_state" in batch:
            return batch["x_state"][:, -1, :]
        return x[:, -1, -self.target_dim :]

    def forward(self, x: torch.Tensor, batch: dict | None = None) -> dict[str, torch.Tensor]:
        hidden_seq, feature = self.backbone.encode(x, batch=batch)
        mode_preds = self.vmd_head(feature)
        y_base = mode_preds.sum(dim=-1)
        last_state_std = self._resolve_last_state_std(x, batch)
        y_hat = self.decoder(y_base, last_state_std=last_state_std)
        y_hat = self.state_mixer(y_hat)
        return {
            "y_hat": y_hat,
            "mode_preds": mode_preds,
            "hidden_seq": hidden_seq,
        }

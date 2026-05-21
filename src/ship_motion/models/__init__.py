"""Step 02 常规基线模型注册表。

这个模块的作用很简单：
1. 统一导出所有 baseline 模型类；
2. 根据配置里的 `model.name` 创建对应模型；
3. 让 train.py / evaluate.py 不需要写一大串 if-else 到处复制。

当前支持的模型：
- persistence
- lstm
- gru
- tcn
- transformer
- lite_xlstm
- ccg_xlstm
- vmd_ccg_xlstm
"""

from __future__ import annotations

from typing import Any

from .gru import GRUForecaster
from .lstm import LSTMForecaster
from .persistence import PersistenceForecaster
from .tcn import TCNForecaster
from .transformer import TransformerForecaster
from .xlstm import CCGXLSTMForecaster, LiteXLSTMForecaster, VMDCCGXLSTMForecaster


def build_model_from_config(model_cfg: dict[str, Any]) -> Any:
    """根据配置创建模型实例。"""
    name = str(model_cfg.get("name", "")).lower()
    if name == "persistence":
        return PersistenceForecaster(
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
        )
    if name == "lstm":
        return LSTMForecaster(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "gru":
        return GRUForecaster(
            input_dim=int(model_cfg["input_dim"]),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "tcn":
        return TCNForecaster(
            input_dim=int(model_cfg["input_dim"]),
            channels=[int(value) for value in model_cfg.get("channels", [64, 128, 128])],
            kernel_size=int(model_cfg.get("kernel_size", 3)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if name == "transformer":
        return TransformerForecaster(
            input_dim=int(model_cfg["input_dim"]),
            d_model=int(model_cfg.get("d_model", 128)),
            nhead=int(model_cfg.get("nhead", 4)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            dim_feedforward=int(model_cfg.get("dim_feedforward", int(model_cfg.get("d_model", 128)) * 4)),
        )
    if name == "lite_xlstm":
        return LiteXLSTMForecaster(
            input_dim=int(model_cfg["input_dim"]),
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            gate_clip=float(model_cfg.get("gate_clip", 5.0)),
        )
    if name == "ccg_xlstm":
        return CCGXLSTMForecaster(
            state_dim=int(model_cfg.get("state_dim", 5)),
            exog_dim=int(model_cfg.get("exog_dim", 6)),
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            context_dim=int(model_cfg.get("context_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            gate_clip=float(model_cfg.get("gate_clip", 5.0)),
        )
    if name == "vmd_ccg_xlstm":
        return VMDCCGXLSTMForecaster(
            state_dim=int(model_cfg.get("state_dim", 5)),
            exog_dim=int(model_cfg.get("exog_dim", 6)),
            d_model=int(model_cfg.get("d_model", 128)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            pred_len=int(model_cfg["pred_len"]),
            target_dim=int(model_cfg["target_dim"]),
            K=int(model_cfg.get("K", 3)),
            context_dim=int(model_cfg.get("context_dim", 64)),
            dropout=float(model_cfg.get("dropout", 0.1)),
            gate_clip=float(model_cfg.get("gate_clip", 5.0)),
            decode_type=str(model_cfg.get("decode_type", "delta")),
            use_state_mixer=bool(model_cfg.get("use_state_mixer", True)),
            state_mixer_hidden_dim=int(model_cfg.get("state_mixer_hidden_dim", 16)),
            state_mixer_dropout=float(model_cfg.get("state_mixer_dropout", 0.0)),
        )
    raise ValueError(f"Unsupported model name: {name!r}")


__all__ = [
    "PersistenceForecaster",
    "LSTMForecaster",
    "GRUForecaster",
    "TCNForecaster",
    "TransformerForecaster",
    "LiteXLSTMForecaster",
    "CCGXLSTMForecaster",
    "VMDCCGXLSTMForecaster",
    "build_model_from_config",
]

from __future__ import annotations

"""项目通用小工具。

这里放的是多个模块都会重复用到的基础能力：
1. 设置随机种子，尽量保证实验可复现；
2. 列出数据目录中的 CSV 文件；
3. 创建输出目录；
4. 读取 YAML / JSON 配置文件。

当前实现依赖成熟工具包而不是手写兜底逻辑：
- YAML 读取使用 PyYAML；
- JSON、路径和目录处理使用 Python 标准库。
"""

import json
import random
from pathlib import Path
from typing import Any

import yaml


def set_seed(seed: int) -> None:
    """统一设置随机种子。

    为什么要做这件事：
    - random 是 Python 自带随机库；
    - numpy 常用于数值计算；
    - torch 常用于深度学习。

    如果三者都设置同一个 seed，那么每次运行程序时，
    随机采样、随机初始化等行为会尽量保持一致，
    更方便复现实验结果和排查问题。
    """
    random.seed(seed)

    try:
        import numpy as np

        # 设置 NumPy 的随机种子。
        np.random.seed(seed)
    except ModuleNotFoundError:
        # 没装 NumPy 时不报错，直接跳过。
        pass

    try:
        import torch

        # 设置 CPU 上的随机种子。
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            # 如果有 GPU，再把所有 GPU 的随机种子也一起设置。
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        # 没装 PyTorch 时也允许继续运行。
        pass


def list_csv_files(data_dir: str | Path) -> list[Path]:
    """列出目录下所有 CSV 文件，并按文件名排序后返回。"""
    path = Path(data_dir)
    if not path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory: {path}")
    return sorted(path.glob("*.csv"))


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在。

    - 如果目录已经存在：什么也不做；
    - 如果目录不存在：自动递归创建。
    """
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件。"""
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    config = loaded or {}
    if isinstance(config, dict):
        config["__config_path__"] = str(source)
    return config


def save_json(obj: Any, path: str | Path) -> None:
    """把对象保存成 JSON 文件。"""
    target = Path(path)
    # 先确保父目录存在，避免“目录不存在导致写文件失败”。
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """从 JSON 文件中读取对象。"""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

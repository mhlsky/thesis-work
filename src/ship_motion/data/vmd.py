from __future__ import annotations

"""VMD 分解与标签缓存构建工具。

本模块只负责两件事：
1. 对目标状态序列做 VMD 分解；
2. 把分解结果缓存成 `.npz`，供后续 Dataset 按窗口切出 `y_modes`。

重要约束：
- VMD 结果在本项目里只作为辅助标签使用；
- 不把 VMD 模态作为模型输入特征，避免测试阶段未来信息泄漏。
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from ship_motion.data.dataset import DEFAULT_CONFIG_PATH, build_datasets
from ship_motion.utils import ensure_dir, list_csv_files, load_yaml


@dataclass(slots=True)
class VMDConfig:
    """VMD 分解参数。"""

    K: int = 3
    alpha: float = 2000.0
    tau: float = 0.0
    DC: bool = False
    init: int = 1
    tol: float = 1e-7
    enabled: bool = True
    lambda_vmd: float = 0.2
    cache_root: str | None = None

    @classmethod
    def from_dict(cls, config: dict[str, Any] | None) -> "VMDConfig":
        """从 YAML 的 `vmd:` 小节构造配置对象。"""
        cfg = config or {}
        return cls(
            K=int(cfg.get("K", 3)),
            alpha=float(cfg.get("alpha", 2000.0)),
            tau=float(cfg.get("tau", 0.0)),
            DC=bool(cfg.get("DC", False)),
            init=int(cfg.get("init", 1)),
            tol=float(cfg.get("tol", 1e-7)),
            enabled=bool(cfg.get("enabled", True)),
            lambda_vmd=float(cfg.get("lambda_vmd", 0.2)),
            cache_root=cfg.get("cache_root"),
        )

    def cache_dir_name(self) -> str:
        """生成与参数绑定的缓存目录名。"""
        alpha_str = f"{self.alpha:g}".replace(".", "_")
        return f"K{self.K}_alpha{alpha_str}"


class VMDCacheBuilder:
    """为多个 CSV 文件构建 VMD 模态缓存。"""

    def __init__(self, config: VMDConfig, target_cols: Sequence[str], cache_root: str | Path) -> None:
        if config.K <= 0:
            raise ValueError("VMDConfig.K must be positive.")
        self.config = config
        self.target_cols = list(target_cols)
        self.cache_root = Path(cache_root)

    def decompose_1d(self, signal: Sequence[float]) -> Any:
        """对单个一维序列做 VMD，返回形状 `[T, K]`。"""
        np = _import_numpy()
        signal_array = np.asarray(signal, dtype=float)
        if signal_array.ndim != 1:
            raise ValueError(f"Expected 1D signal for VMD, got shape {signal_array.shape}")
        if signal_array.size == 0:
            raise ValueError("Signal for VMD must not be empty.")
        original_length = int(signal_array.shape[0])

        # vmdpy 遇到奇数长度序列时会直接裁掉最后一个点。
        # 这里主动补一个末尾值，保证输出仍能与原始时间轴一一对齐。
        if original_length % 2 == 1:
            signal_array = np.pad(signal_array, (0, 1), mode="edge")

        vmd_fn = _import_vmd_function()
        modes, _, _ = vmd_fn(
            signal_array,
            self.config.alpha,
            self.config.tau,
            self.config.K,
            int(self.config.DC),
            self.config.init,
            self.config.tol,
        )
        modes = np.asarray(modes, dtype=float)
        if modes.ndim != 2:
            raise ValueError(f"Unexpected VMD output shape: {modes.shape}")
        if modes.shape[0] == self.config.K:
            # vmdpy 常见输出是 [K, T]，这里统一转成 [T, K]。
            modes = modes.transpose(1, 0)
        if modes.shape[1] != self.config.K:
            raise ValueError(
                f"Unexpected VMD mode count: expected {self.config.K}, got {modes.shape[1]}"
            )
        return modes[:original_length]

    def build_file_cache(self, csv_path: str | Path, split_name: str) -> Path:
        """读取一个 CSV，分解目标列并写出 `.npz` 缓存。"""
        np = _import_numpy()
        csv_path = Path(csv_path)
        target_matrix = _read_target_matrix(csv_path, self.target_cols)

        decomposed_cols = [self.decompose_1d(target_matrix[:, idx]) for idx in range(target_matrix.shape[1])]
        modes = np.stack(decomposed_cols, axis=1)
        # 现在的形状是 [T, target_dim, K]，正好便于 Dataset 直接按时间切片。

        split_dir = ensure_dir(self.cache_root / split_name)
        cache_path = split_dir / f"{csv_path.stem}.npz"
        np.savez_compressed(
            cache_path,
            modes=modes.astype(np.float32),
            target_cols=np.asarray(self.target_cols),
            K=np.asarray(self.config.K, dtype=np.int32),
        )
        return cache_path

    def build_split_cache(
        self,
        data_dir: str | Path,
        split_name: str,
        max_files: int | None = None,
    ) -> list[Path]:
        """为一个数据划分批量生成缓存。"""
        files = list_csv_files(data_dir)
        if max_files is not None:
            files = files[: int(max_files)]
        if not files:
            raise FileNotFoundError(f"No CSV files found for split {split_name}: {data_dir}")

        built_paths: list[Path] = []
        for file in files:
            cache_path = self.build_file_cache(file, split_name=split_name)
            built_paths.append(cache_path)
            print(f"[VMD] built {split_name}: {file.name} -> {cache_path}")
        return built_paths


def build_vmd_cache_from_config(
    config: dict[str, Any],
    smoke: bool = False,
    cache_root_override: str | Path | None = None,
) -> dict[str, list[Path]]:
    """根据项目配置为四个数据划分生成缓存。"""
    data_cfg = _resolve_data_paths(config)
    vmd_cfg = VMDConfig.from_dict(config.get("vmd"))
    if not vmd_cfg.enabled:
        raise ValueError("VMD cache build requested, but config.vmd.enabled is False.")

    max_files = 1 if smoke else data_cfg.get("max_files")
    cache_root = Path(cache_root_override) if cache_root_override is not None else _resolve_cache_root(config, vmd_cfg)
    builder = VMDCacheBuilder(vmd_cfg, data_cfg["target_cols"], cache_root)

    built = {
        "train": builder.build_split_cache(data_cfg["train_dir"], "train", max_files=max_files),
        "validation": builder.build_split_cache(data_cfg["val_dir"], "validation", max_files=max_files),
        "routine_test": builder.build_split_cache(
            data_cfg["routine_test_dir"], "routine_test", max_files=max_files
        ),
        "ood_test": builder.build_split_cache(data_cfg["ood_test_dir"], "ood_test", max_files=max_files),
    }
    return built


def run_smoke_test(
    config_path: str | Path = DEFAULT_CONFIG_PATH.parent / "vmd_ccg_xlstm.yaml",
    cache_root_override: str | Path | None = None,
) -> None:
    """执行 Step 03 的轻量 smoke test。"""
    config = load_yaml(config_path)
    smoke_cache_root = cache_root_override or Path("outputs/cache/vmd_smoke") / VMDConfig.from_dict(
        config.get("vmd")
    ).cache_dir_name()
    build_vmd_cache_from_config(config, smoke=True, cache_root_override=smoke_cache_root)

    # 验证 Dataset 能否把 y_modes 一并返回。
    dataset_config = dict(config)
    dataset_config["vmd"] = dict(dataset_config.get("vmd", {}))
    dataset_config["data"] = dict(dataset_config["data"])
    dataset_config["vmd"]["enabled"] = True
    dataset_config["vmd"]["cache_root"] = str(smoke_cache_root)
    dataset_config["data"]["max_files"] = 1
    dataset_config["data"]["max_windows_per_file"] = 8
    bundle = build_datasets(dataset_config, smoke=False)
    sample = bundle["train"][0]

    print(f"[Smoke] VMD cache root: {Path(smoke_cache_root).resolve()}")
    print(f"[Smoke] train windows: {len(bundle['train'])}")
    print(f"[Smoke] y shape: {_shape(sample['y'])}")
    print(f"[Smoke] y_modes shape: {_shape(sample['y_modes'])}")
    print(f"[Smoke] sample file: {sample['file']}")
    print(f"[Smoke] sample start: {sample['start']}")


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。"""
    default_config = DEFAULT_CONFIG_PATH.parent / "vmd_ccg_xlstm.yaml"
    parser = argparse.ArgumentParser(description="Build VMD auxiliary-label cache for ship motion data.")
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        run_smoke_test(config_path=args.config, cache_root_override=args.cache_root)
        return

    config = load_yaml(args.config)
    built = build_vmd_cache_from_config(config, smoke=False, cache_root_override=args.cache_root)
    for split_name, paths in built.items():
        print(f"[VMD] {split_name}: built {len(paths)} cache files")


def _read_target_matrix(csv_path: Path, target_cols: Sequence[str]) -> Any:
    """按 target_cols 顺序读取单个 CSV 的目标矩阵。"""
    np = _import_numpy()
    rows: list[list[float]] = []

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [col for col in target_cols if col not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        for row_idx, row in enumerate(reader, start=2):
            try:
                rows.append([float(row[col]) for col in target_cols])
            except ValueError as exc:
                raise ValueError(f"Bad numeric value in {csv_path}:{row_idx}") from exc

    if not rows:
        raise ValueError(f"CSV file is empty or has no data rows: {csv_path}")
    return np.asarray(rows, dtype=float)


def _resolve_data_paths(config: dict[str, Any]) -> dict[str, Any]:
    """解析配置中的数据目录。"""
    data_cfg = dict(config["data"])
    config_path = config.get("__config_path__")
    if config_path is None:
        return data_cfg

    config_dir = Path(config_path).resolve().parent
    repo_root = DEFAULT_CONFIG_PATH.parent.parent
    for key in ("train_dir", "val_dir", "routine_test_dir", "ood_test_dir"):
        data_cfg[key] = str(_resolve_path(data_cfg[key], config_dir, repo_root))
    return data_cfg


def _resolve_cache_root(config: dict[str, Any], vmd_cfg: VMDConfig) -> Path:
    """解析缓存根目录。"""
    config_path = config.get("__config_path__")
    configured = vmd_cfg.cache_root or f"outputs/cache/vmd/{vmd_cfg.cache_dir_name()}"
    if config_path is None:
        return Path(configured).resolve()
    config_dir = Path(config_path).resolve().parent
    repo_root = DEFAULT_CONFIG_PATH.parent.parent
    return _resolve_path(configured, config_dir, repo_root)


def _resolve_path(path_value: str | Path, config_dir: Path, repo_root: Path) -> Path:
    """解析相对路径，避免依赖当前工作目录。"""
    path = Path(path_value)
    if path.is_absolute():
        return path
    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative
    return (repo_root / path).resolve()


def _import_numpy() -> Any:
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("NumPy is required for VMD cache building. Please run `uv sync`.") from exc
    return np


def _import_vmd_function() -> Any:
    try:
        from vmdpy import VMD
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("vmdpy is required for Step 03. Please run `uv sync`.") from exc
    return VMD


def _shape(value: Any) -> Any:
    """打印时统一查看 shape，兼容 numpy / torch / 列表。"""
    if hasattr(value, "shape"):
        return value.shape
    if hasattr(value, "tolist"):
        value = value.tolist()
    shape: list[int] = []
    current = value
    while isinstance(current, list):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


if __name__ == "__main__":
    main()

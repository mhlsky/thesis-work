from __future__ import annotations

"""数据集构建逻辑。

这个文件主要负责两件事：
1. 读取多份船舶 CSV 数据；
2. 把长时间序列切成“历史输入窗口 + 未来预测窗口”。

如果你把模型训练理解成“喂很多训练样本”，
那么本文件就是负责“生产样本”的地方。
"""

import argparse
import csv
import sys
from copy import deepcopy
from itertools import islice
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ship_motion.data.scaler import StandardScaler
from ship_motion.utils import ensure_dir, list_csv_files, load_yaml, set_seed


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "base.yaml"


class ArrayTensor:
    """没有安装 PyTorch 时使用的“轻量张量替代品”。

    作用：
    - 让数据对象至少有 `.shape`、`tolist()` 这些常见接口；
    - 这样 smoke test 之类的小测试，即使没装 torch 也能跑通。
    """

    def __init__(self, data: Any) -> None:
        self.data = data
        self.shape = _shape_of_nested(data)

    def __getitem__(self, item: Any) -> Any:
        return self.data[item]

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return f"ArrayTensor(shape={self.shape})"

    def tolist(self) -> Any:
        return self.data


class ShipWindowDataset:
    """把原始时间序列切成滑动窗口样本的数据集。

    一个样本大致长这样：
    - x: 过去 seq_len 步的输入特征
    - y: 未来 pred_len 步的目标状态

    例如：
    - seq_len = 128
    - pred_len = 10

    那么一条长轨迹会被切成很多段：
    前 128 步作为输入，后 10 步作为预测目标。
    再往前滑动 stride 个时间步，就得到下一个样本。
    """

    def __init__(
        self,
        files: Sequence[str | Path],
        scaler: StandardScaler,
        input_cols: Sequence[str],
        target_cols: Sequence[str],
        exog_cols: Sequence[str],
        state_cols: Sequence[str],
        seq_len: int,
        pred_len: int,
        stride: int = 1,
        max_windows_per_file: int | None = None,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")

        # 把文件路径统一转成 Path，后面处理更方便。
        self.files = [Path(file) for file in files]
        self.scaler = scaler
        self.input_cols = list(input_cols)
        self.target_cols = list(target_cols)
        self.exog_cols = list(exog_cols)
        self.state_cols = list(state_cols)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.max_windows_per_file = max_windows_per_file
        # 记录“外生变量列”和“状态列”在 input_cols 里的位置，
        # 方便后面从 x 中快速切出子特征。
        self.exog_indices = _column_indices(self.input_cols, self.exog_cols)
        self.state_indices = _column_indices(self.input_cols, self.state_cols)

        # _file_cache: 保存每个文件解析后的二维数组，避免反复读盘。
        self._file_cache: list[dict[str, Any]] = []
        # _index: 保存“第几个文件 + 从哪一行开始切窗口”。
        self._index: list[tuple[int, int]] = []
        self._build_index()

    def __len__(self) -> int:
        """数据集总共有多少个窗口样本。"""
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """取出一个训练样本。

        返回内容包括：
        - x: 标准化后的历史输入
        - x_exog: 从 x 中切出来的外生变量
        - x_state: 从 x 中切出来的状态变量
        - y: 标准化后的未来目标
        - y_raw: 未标准化的未来目标
        - last_state_raw: 输入窗口最后一个时刻的真实状态
        - file/start: 样本来自哪个文件、从第几行开始
        """
        file_idx, start = self._index[index]
        cached = self._file_cache[file_idx]

        # 过去 seq_len 步作为模型输入。
        x_raw = cached["x"][start : start + self.seq_len]
        # 紧随其后的 pred_len 步作为监督信号。
        y_raw = cached["y"][start + self.seq_len : start + self.seq_len + self.pred_len]
        # 记录输入窗口最后一个真实状态，很多序列模型会拿它做解码起点。
        last_state_raw = cached["y"][start + self.seq_len - 1]

        # 训练通常使用标准化后的数据，数值更稳定。
        x = self.scaler.transform_x(x_raw)
        y = self.scaler.transform_y(y_raw)

        return {
            "x": _to_tensor(x),
            "x_exog": _to_tensor(_take_columns(x, self.exog_indices)),
            "x_state": _to_tensor(_take_columns(x, self.state_indices)),
            "y": _to_tensor(y),
            "y_raw": _to_tensor(y_raw),
            "last_state_raw": _to_tensor(last_state_raw),
            "file": str(cached["file"]),
            "start": start,
        }

    def _build_index(self) -> None:
        """预先建立“所有可用窗口”的索引表。"""
        for file_idx, file in enumerate(self.files):
            # 每个 CSV 只读一次，读取结果缓存下来。
            cached = _read_csv_arrays(file, self.input_cols, self.target_cols)
            self._file_cache.append(cached)

            row_count = len(cached["x"])
            # 一个窗口需要 seq_len + pred_len 行数据。
            # max_start 表示允许的最后一个起点位置。
            max_start = row_count - self.seq_len - self.pred_len
            if max_start < 0:
                # 轨迹太短，不足以切出一个完整样本，直接跳过。
                continue

            starts = range(0, max_start + 1, self.stride)
            if self.max_windows_per_file is not None:
                # smoke test 时可限制每个文件最多切几个窗口，缩短运行时间。
                starts = islice(starts, self.max_windows_per_file)
            for start in starts:
                self._index.append((file_idx, start))


def build_datasets(config: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    """根据配置构建 train / val / test 数据集。

    返回一个 bundle 字典，里面同时放：
    - 原始文件列表；
    - 训练集统计得到的 scaler；
    - 各个划分对应的数据集对象。
    """
    cfg = deepcopy(config)
    config_path = cfg.get("__config_path__")
    data_cfg = _resolve_data_config_paths(cfg["data"], config_path)
    train_cfg = cfg.get("train", {})

    if smoke:
        # 冒烟测试只取很少的数据，确保流程可跑通即可。
        data_cfg["max_files"] = 2
        data_cfg["max_windows_per_file"] = 50

    set_seed(int(train_cfg.get("seed", 42)))

    max_files = data_cfg.get("max_files")
    train_files = _limited_files(data_cfg["train_dir"], max_files)
    val_files = _limited_files(data_cfg["val_dir"], max_files)
    routine_test_files = _limited_files(data_cfg["routine_test_dir"], max_files)
    ood_test_files = _limited_files(data_cfg["ood_test_dir"], max_files)

    # 只用训练集拟合标准化参数，避免“数据泄漏”。
    scaler = StandardScaler().fit_csv_files(
        train_files,
        input_cols=data_cfg["input_cols"],
        target_cols=data_cfg["target_cols"],
    )

    # 这些参数对 train / val / test 都是共用的。
    common_kwargs = {
        "scaler": scaler,
        "input_cols": data_cfg["input_cols"],
        "target_cols": data_cfg["target_cols"],
        "exog_cols": data_cfg["exog_cols"],
        "state_cols": data_cfg["state_cols"],
        "seq_len": int(data_cfg["seq_len"]),
        "pred_len": int(data_cfg["pred_len"]),
        "stride": int(data_cfg.get("stride", 1)),
        "max_windows_per_file": data_cfg.get("max_windows_per_file"),
    }

    return {
        "train_files": train_files,
        "val_files": val_files,
        "routine_test_files": routine_test_files,
        "ood_test_files": ood_test_files,
        "scaler": scaler,
        "train": ShipWindowDataset(train_files, **common_kwargs),
        "val": ShipWindowDataset(val_files, **common_kwargs),
        "routine_test": ShipWindowDataset(routine_test_files, **common_kwargs),
        "ood_test": ShipWindowDataset(ood_test_files, **common_kwargs),
    }


def run_smoke_test(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """跑一个非常轻量的冒烟测试。

    目的不是训练模型，而是快速检查：
    - 配置能否读取；
    - CSV 能否正常解析；
    - 数据窗口能否正确切分；
    - 标准化器能否正常保存。
    """
    config = load_yaml(config_path)
    bundle = build_datasets(config, smoke=True)
    sample = bundle["train"][0]
    batch = _simple_batch([bundle["train"][idx] for idx in range(min(4, len(bundle["train"])))])

    output_dir = ensure_dir(Path(config.get("output_dir", "outputs")) / "smoke_test")
    scaler_path = output_dir / "scaler.json"
    bundle["scaler"].save(scaler_path)

    print(f"train files: {len(bundle['train_files'])}")
    print(f"val files: {len(bundle['val_files'])}")
    print(f"routine test files: {len(bundle['routine_test_files'])}")
    print(f"OOD test files: {len(bundle['ood_test_files'])}")
    print(f"train windows: {len(bundle['train'])}")
    print(f"val windows: {len(bundle['val'])}")
    print(f"x shape: {_shape(sample['x'])}")
    print(f"x_exog shape: {_shape(sample['x_exog'])}")
    print(f"x_state shape: {_shape(sample['x_state'])}")
    print(f"y shape: {_shape(sample['y'])}")
    print(f"y_raw shape: {_shape(sample['y_raw'])}")
    print(f"last_state_raw shape: {_shape(sample['last_state_raw'])}")
    print(f"batch x shape: {_shape(batch['x'])}")
    print(f"batch y shape: {_shape(batch['y'])}")
    print(f"sample file: {sample['file']}")
    print(f"sample start: {sample['start']}")
    print(f"scaler saved to {scaler_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。

    用法示例：
    - python -m ship_motion.data.dataset --config configs/base.yaml --smoke
    """
    parser = argparse.ArgumentParser(description="Ship motion data pipeline smoke test.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    if args.smoke:
        run_smoke_test(args.config)
    else:
        config = load_yaml(args.config)
        bundle = build_datasets(config, smoke=False)
        print(f"train files: {len(bundle['train_files'])}")
        print(f"val files: {len(bundle['val_files'])}")
        print(f"train windows: {len(bundle['train'])}")
        print(f"val windows: {len(bundle['val'])}")


def _resolve_data_config_paths(
    data_cfg: dict[str, Any],
    config_path: str | Path | None,
) -> dict[str, Any]:
    """把配置中的相对目录解析成绝对路径。

    解析规则：
    - 优先相对配置文件所在目录；
    - 如果该路径不存在，再尝试相对仓库根目录。
    """
    resolved = deepcopy(data_cfg)
    if config_path is None:
        return resolved

    config_dir = Path(config_path).resolve().parent
    repo_root = DEFAULT_CONFIG_PATH.parent.parent
    for key in ("train_dir", "val_dir", "routine_test_dir", "ood_test_dir"):
        value = resolved.get(key)
        if value is None:
            continue
        resolved[key] = str(_resolve_config_path(value, config_dir, repo_root))
    return resolved


def _resolve_config_path(path_value: str | Path, config_dir: Path, repo_root: Path) -> Path:
    """解析配置路径，避免依赖进程当前工作目录。"""
    path = Path(path_value)
    if path.is_absolute():
        return path

    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative

    return (repo_root / path).resolve()


def _read_csv_arrays(
    file: str | Path,
    input_cols: Sequence[str],
    target_cols: Sequence[str],
) -> dict[str, Any]:
    """把单个 CSV 文件读成两个二维数组。

    - x_rows: 输入特征矩阵
    - y_rows: 目标状态矩阵
    """
    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    # 去重后得到“这个 CSV 至少必须包含哪些列”。
    required_cols = list(dict.fromkeys([*input_cols, *target_cols]))

    with Path(file).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        _validate_columns(reader.fieldnames, required_cols, file)
        for row_idx, row in enumerate(reader, start=2):
            x_rows.append(_read_float_values(row, input_cols, file, row_idx))
            y_rows.append(_read_float_values(row, target_cols, file, row_idx))

    return {"file": Path(file), "x": x_rows, "y": y_rows}


def _limited_files(data_dir: str | Path, max_files: int | None) -> list[Path]:
    """读取目录中的 CSV 文件，并按需要截断数量。"""
    files = list_csv_files(data_dir)
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def _column_indices(all_cols: Sequence[str], selected_cols: Sequence[str]) -> list[int]:
    """把列名列表转换成列下标列表。"""
    missing = [col for col in selected_cols if col not in all_cols]
    if missing:
        raise ValueError(f"Columns are not in input_cols: {missing}")
    return [all_cols.index(col) for col in selected_cols]


def _take_columns(rows: Sequence[Sequence[float]], indices: Sequence[int]) -> list[list[float]]:
    """从二维数组中按下标抽取指定列。"""
    return [[row[idx] for idx in indices] for row in rows]


def _to_tensor(data: Any) -> Any:
    """优先转成 torch.Tensor；如果没装 torch，则退化为 ArrayTensor。"""
    torch = _try_import_torch()
    if torch is not None:
        return torch.tensor(data, dtype=torch.float32)
    return ArrayTensor(data)


def _shape(value: Any) -> Any:
    """统一获取对象形状，兼容 Tensor 和普通嵌套列表。"""
    if hasattr(value, "shape"):
        return value.shape
    return _shape_of_nested(value)


def _shape_of_nested(value: Any) -> tuple[int, ...]:
    """根据嵌套列表结构推断 shape。

    例如：
    - [[1, 2], [3, 4]] -> (2, 2)
    - [[[1]], [[2]]] -> (2, 1, 1)
    """
    shape: list[int] = []
    current = value
    while isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return tuple(shape)


def _simple_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """手工拼一个极简 batch。

    这里只在 smoke test 中使用，
    目的只是验证多个样本能否堆叠成 batch。
    """
    return {
        "x": _to_tensor([_as_list(sample["x"]) for sample in samples]),
        "y": _to_tensor([_as_list(sample["y"]) for sample in samples]),
    }


def _as_list(value: Any) -> Any:
    """把 Tensor / ArrayTensor / 原生列表统一转成可嵌套的列表结构。"""
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _validate_columns(
    fieldnames: Sequence[str] | None,
    required_cols: Sequence[str],
    file: str | Path,
) -> None:
    """检查 CSV 是否包含所需列。"""
    if fieldnames is None:
        raise ValueError(f"CSV has no header: {file}")
    missing = [col for col in required_cols if col not in fieldnames]
    if missing:
        raise ValueError(f"Missing columns in {file}: {missing}")


def _read_float_values(
    row: dict[str, str],
    cols: Sequence[str],
    file: str | Path,
    row_idx: int,
) -> list[float]:
    """按列名顺序读取一行中的多个数值。"""
    values: list[float] = []
    for col in cols:
        raw = row.get(col, "")
        try:
            values.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Bad numeric value in {file}:{row_idx}, column {col}: {raw!r}") from exc
    return values


def _try_import_torch() -> Any:
    """尝试导入 PyTorch，没装就返回 None。"""
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    main()

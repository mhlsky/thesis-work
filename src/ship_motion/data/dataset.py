from __future__ import annotations

"""数据集构建逻辑。

这个文件主要负责两件事：
1. 读取多份船舶 CSV 数据；
2. 把长时间序列切成“历史输入窗口 + 未来预测窗口”。

如果你把模型训练理解成“喂很多训练样本”，
那么本文件就是负责“生产样本”的地方。

建议把本模块想成三层：
1. 最外层 `build_datasets`：根据配置搭建 train / val / test 数据集；
2. 中间层 `ShipWindowDataset`：把单条长时间序列切成很多滑动窗口样本；
3. 底层辅助函数：负责读 CSV、解析路径、抽取列、把列表转成张量等杂务。
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
        """保存原始嵌套数据，并提前推断 shape。"""
        self.data = data
        self.shape = _shape_of_nested(data)

    def __getitem__(self, item: Any) -> Any:
        """支持像列表或张量一样按下标取值。"""
        return self.data[item]

    def __len__(self) -> int:
        """返回第一维长度，行为尽量贴近常见张量对象。"""
        return len(self.data)

    def __repr__(self) -> str:
        """打印时优先展示 shape，便于调试。"""
        return f"ArrayTensor(shape={self.shape})"

    def tolist(self) -> Any:
        """与 torch.Tensor.tolist() 保持类似接口。"""
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
        vmd_cache_dir: str | Path | None = None,
        vmd_enabled: bool = False,
        vmd_K: int | None = None,
    ) -> None:
        """初始化一个“滑动窗口数据集”。

        参数可以先这样理解：
        - files: 数据来源，每个文件通常是一条或一批时间序列；
        - scaler: 已经用训练集拟合好的标准化器；
        - input_cols: 模型输入需要哪些列；
        - target_cols: 模型监督目标需要哪些列；
        - exog_cols: 输入里的外生变量列；
        - state_cols: 输入里的状态变量列；
        - seq_len: 每个样本向后看多少步历史；
        - pred_len: 每个样本向前预测多少步未来；
        - stride: 滑动窗口每次前进多少步；
        - max_windows_per_file: 每个文件最多切多少个窗口，常用于 smoke test。
        """
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
        self.vmd_enabled = vmd_enabled
        self.vmd_K = vmd_K
        self.vmd_cache_dir = Path(vmd_cache_dir) if vmd_cache_dir is not None else None
        # 记录“外生变量列”和“状态列”在 input_cols 里的位置，
        # 方便后面从 x 中快速切出子特征。
        self.exog_indices = _column_indices(self.input_cols, self.exog_cols)
        self.state_indices = _column_indices(self.input_cols, self.state_cols)

        if self.vmd_enabled:
            if self.vmd_cache_dir is None:
                raise ValueError("vmd_cache_dir must be provided when vmd_enabled=True.")
            if self.vmd_K is None or self.vmd_K <= 0:
                raise ValueError("vmd_K must be a positive integer when vmd_enabled=True.")

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
            **(
                {
                    # y_modes 保持原始物理尺度，不做标准化。
                    # 原因是后续 VMD 辅助监督通常直接对应真实模态分量，
                    # 而不是当前训练框架里的标准化 y。
                    "y_modes": _to_tensor(
                        cached["y_modes"][start + self.seq_len : start + self.seq_len + self.pred_len]
                    ),
                }
                if self.vmd_enabled
                else {}
            ),
        }

    def _build_index(self) -> None:
        """预先建立“所有可用窗口”的索引表。"""
        for file_idx, file in enumerate(self.files):
            # 每个 CSV 只读一次，读取结果缓存下来。
            cached = _read_csv_arrays(file, self.input_cols, self.target_cols)
            if self.vmd_enabled:
                cached["y_modes"] = _load_vmd_modes(
                    file=file,
                    cache_dir=self.vmd_cache_dir,
                    target_cols=self.target_cols,
                    expected_length=len(cached["y"]),
                    expected_k=int(self.vmd_K),
                )
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

    这个函数通常是“数据模块总入口”：
    - 先读取并修正配置里的路径；
    - 再收集 train / val / test 各自对应的 CSV 文件；
    - 接着只用训练集拟合标准化器；
    - 最后为每个数据划分创建一个 ShipWindowDataset。
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

    vmd_cfg = cfg.get("vmd", {})
    vmd_enabled = bool(vmd_cfg.get("enabled", False))
    if vmd_enabled:
        vmd_cache_root = _resolve_optional_config_path(
            vmd_cfg.get("cache_root") or _default_vmd_cache_root(vmd_cfg),
            config_path,
        )
        common_kwargs.update(
            {
                "vmd_enabled": True,
                "vmd_K": int(vmd_cfg["K"]),
            }
        )
    else:
        vmd_cache_root = None

    return {
        "train_files": train_files,
        "val_files": val_files,
        "routine_test_files": routine_test_files,
        "ood_test_files": ood_test_files,
        "scaler": scaler,
        "train": ShipWindowDataset(
            train_files,
            **common_kwargs,
            vmd_cache_dir=(vmd_cache_root / "train") if vmd_enabled else None,
        ),
        "val": ShipWindowDataset(
            val_files,
            **common_kwargs,
            vmd_cache_dir=(vmd_cache_root / "validation") if vmd_enabled else None,
        ),
        "routine_test": ShipWindowDataset(
            routine_test_files,
            **common_kwargs,
            vmd_cache_dir=(vmd_cache_root / "routine_test") if vmd_enabled else None,
        ),
        "ood_test": ShipWindowDataset(
            ood_test_files,
            **common_kwargs,
            vmd_cache_dir=(vmd_cache_root / "ood_test") if vmd_enabled else None,
        ),
    }


def run_smoke_test(config_path: str | Path = DEFAULT_CONFIG_PATH) -> None:
    """跑一个非常轻量的冒烟测试。

    目的不是训练模型，而是快速检查：
    - 配置能否读取；
    - CSV 能否正常解析；
    - 数据窗口能否正确切分；
    - 标准化器能否正常保存。

    你可以把 smoke test 理解成“点火试车”：
    不追求完整训练，只确认数据管线能不能跑起来。
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
    if "y_modes" in sample:
        print(f"y_modes shape: {_shape(sample['y_modes'])}")
    print(f"batch x shape: {_shape(batch['x'])}")
    print(f"batch y shape: {_shape(batch['y'])}")
    print(f"sample file: {sample['file']}")
    print(f"sample start: {sample['start']}")
    print(f"scaler saved to {scaler_path}")


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。

    用法示例：
    - python -m ship_motion.data.dataset --config configs/base.yaml --smoke

    这个入口主要是为了命令行快速验证数据流程，
    不是完整训练脚本。
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
    """解析配置路径，避免依赖进程当前工作目录。

    为什么要单独做这一步：
    - 如果直接写相对路径，程序从不同目录启动时可能找不到文件；
    - 这里统一把路径变成绝对路径，后续逻辑就更稳定。
    """
    path = Path(path_value)
    if path.is_absolute():
        return path

    config_relative = (config_dir / path).resolve()
    if config_relative.exists():
        return config_relative

    return (repo_root / path).resolve()


def _resolve_optional_config_path(
    path_value: str | Path | None,
    config_path: str | Path | None,
) -> Path | None:
    """解析可选配置路径。

    与 `_resolve_config_path` 的区别是：
    - 允许传入 None；
    - 便于处理 VMD 缓存目录这类“可开可关”的路径字段。
    """
    if path_value is None:
        return None
    if config_path is None:
        return Path(path_value).resolve()
    config_dir = Path(config_path).resolve().parent
    repo_root = DEFAULT_CONFIG_PATH.parent.parent
    return _resolve_config_path(path_value, config_dir, repo_root)


def _default_vmd_cache_root(vmd_cfg: dict[str, Any]) -> str:
    """根据 VMD 参数生成默认缓存目录名。"""
    k = int(vmd_cfg.get("K", 3))
    alpha = vmd_cfg.get("alpha", 2000)
    alpha_str = str(alpha).replace(".", "_")
    return f"outputs/cache/vmd/K{k}_alpha{alpha_str}"


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


def _load_vmd_modes(
    file: str | Path,
    cache_dir: Path | None,
    target_cols: Sequence[str],
    expected_length: int,
    expected_k: int,
) -> Any:
    """读取单个 CSV 对应的 VMD 模态缓存。

    缓存文件名与 CSV stem 对齐，例如：
    - `train/xxx.csv`
    - `outputs/cache/vmd/.../train/xxx.npz`
    """
    if cache_dir is None:
        raise ValueError("cache_dir must not be None when loading VMD modes.")

    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "NumPy is required for loading VMD cache. Please run `uv sync` first."
        ) from exc

    cache_path = cache_dir / f"{Path(file).stem}.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"VMD cache not found for {file}. Expected cache file: {cache_path}"
        )

    with np.load(cache_path, allow_pickle=False) as loaded:
        modes = loaded["modes"]
        cached_target_cols = loaded["target_cols"].tolist()
        cached_k = int(loaded["K"])

    if list(cached_target_cols) != list(target_cols):
        raise ValueError(
            f"VMD cache target_cols mismatch for {cache_path}: "
            f"expected {list(target_cols)}, got {list(cached_target_cols)}"
        )
    if modes.shape[0] != expected_length:
        raise ValueError(
            f"VMD cache length mismatch for {cache_path}: "
            f"expected {expected_length}, got {modes.shape[0]}"
        )
    if modes.shape[1] != len(target_cols):
        raise ValueError(
            f"VMD cache target_dim mismatch for {cache_path}: "
            f"expected {len(target_cols)}, got {modes.shape[1]}"
        )
    if cached_k != expected_k or modes.shape[2] != expected_k:
        raise ValueError(
            f"VMD cache K mismatch for {cache_path}: expected {expected_k}, got {cached_k}"
        )
    return modes.tolist()


def _limited_files(data_dir: str | Path, max_files: int | None) -> list[Path]:
    """读取目录中的 CSV 文件，并按需要截断数量。

    这个函数常用于两种场景：
    - 正常训练时读取目录下全部 CSV；
    - smoke test 时只取前几个文件，加快验证速度。
    """
    files = list_csv_files(data_dir)
    if max_files is not None:
        files = files[: int(max_files)]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    return files


def _column_indices(all_cols: Sequence[str], selected_cols: Sequence[str]) -> list[int]:
    """把列名列表转换成列下标列表。

    例如：
    - all_cols = [a, b, c, d]
    - selected_cols = [b, d]
    - 返回 [1, 3]

    这样后面就能更快地按位置切列，而不是反复按列名查找。
    """
    missing = [col for col in selected_cols if col not in all_cols]
    if missing:
        raise ValueError(f"Columns are not in input_cols: {missing}")
    return [all_cols.index(col) for col in selected_cols]


def _take_columns(rows: Sequence[Sequence[float]], indices: Sequence[int]) -> list[list[float]]:
    """从二维数组中按下标抽取指定列。

    输入通常形如 [time, feature]，
    输出仍是二维数组，只是保留了部分特征列。
    """
    return [[row[idx] for idx in indices] for row in rows]


def _to_tensor(data: Any) -> Any:
    """优先转成 torch.Tensor；如果没装 torch，则退化为 ArrayTensor。

    这样写的好处是：
    - 装了 PyTorch 时，训练代码可以直接吃张量；
    - 没装 PyTorch 时，最基础的数据流程和 smoke test 仍然能跑。
    """
    torch = _try_import_torch()
    if torch is not None:
        return torch.tensor(data, dtype=torch.float32)
    return ArrayTensor(data)


def _shape(value: Any) -> Any:
    """统一获取对象形状，兼容 Tensor 和普通嵌套列表。

    这是一个小工具函数，主要为了打印调试信息时更统一。
    """
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
    """把 Tensor / ArrayTensor / 原生列表统一转成可嵌套的列表结构。

    这样 `_simple_batch` 就不用关心输入到底来自 torch 还是纯 Python。
    """
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
    """尝试导入 PyTorch，没装就返回 None。

    这是一个“软依赖”设计：
    - 有 torch 时走真实张量逻辑；
    - 没 torch 时走降级逻辑；
    - 这样数据模块不会因为缺少深度学习框架而完全不可用。
    """
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


if __name__ == "__main__":
    main()

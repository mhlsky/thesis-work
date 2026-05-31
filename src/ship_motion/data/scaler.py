from __future__ import annotations

"""基于 scikit-learn 的标准化工具封装。

这个模块负责项目里的数据标准化工作，核心职责是：
1. 根据训练集 CSV 统计输入特征和目标变量的均值、标准差；
2. 在训练和评估时对 x / y 做标准化；
3. 在预测输出后把 y 从标准化空间还原回真实物理量；
4. 把标准化参数保存成 JSON，便于后续加载复用。

之所以保留这一层项目封装，而不是在上层直接调用
`sklearn.preprocessing.StandardScaler`，是因为这里还要处理：
- 输入列和目标列分开统计；
- 嵌套列表、序列窗口和 torch.Tensor 的统一接口；
- 项目自己的 `scaler.json` 保存 / 加载格式。

如果把本模块一句话讲清楚：
它负责把“原始物理量”转换成“更适合模型学习的数值空间”，
并在需要时再转换回来。
"""

import csv
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from ship_motion.utils import load_json, save_json


class StandardScaler:
    """项目标准化器，对输入 x 和目标 y 分别维护一套统计量。

    这里不要把它理解成“一个普通的 sklearn.StandardScaler 对象”。
    它其实做了三层事情：
    1. 对输入特征 x 统计一套 mean/std；
    2. 对预测目标 y 再单独统计一套 mean/std；
    3. 对外暴露统一接口，让 list / numpy / torch.Tensor 都能走同一套标准化逻辑。

    为什么要拆成 x / y 两套：
    - 输入列和目标列通常不是同一批字段；
    - 即使字段数一样，数值分布也可能完全不同；
    - 训练时模型吃的是标准化后的 x，监督信号也是标准化后的 y；
    - 但推理输出后，又只需要把 y 反标准化回真实物理量。
    """

    def __init__(
        self,
        input_cols: Sequence[str] | None = None,
        target_cols: Sequence[str] | None = None,
        x_mean: Sequence[float] | None = None,
        x_std: Sequence[float] | None = None,
        y_mean: Sequence[float] | None = None,
        y_std: Sequence[float] | None = None,
    ) -> None:
        """创建项目标准化器对象。

        一开始这些统计量可以是空的；
        真正的均值和标准差通常在 `fit_csv_files()` 之后才会得到。

        如果传入了 x_mean / x_std / y_mean / y_std，
        则表示这是一个“从已保存参数恢复出来”的 scaler。
        """
        self.input_cols = list(input_cols or [])
        self.target_cols = list(target_cols or [])
        self.x_mean = list(x_mean or [])
        self.x_std = list(x_std or [])
        self.y_mean = list(y_mean or [])
        self.y_std = list(y_std or [])
        self._x_scaler = _restore_sklearn_scaler(self.x_mean, self.x_std)
        self._y_scaler = _restore_sklearn_scaler(self.y_mean, self.y_std)

    def fit_csv_files(
        self,
        files: Iterable[str | Path],
        input_cols: Sequence[str],
        target_cols: Sequence[str],
    ) -> "StandardScaler":
        """用训练集 CSV 计算 x / y 两套均值和标准差。

        注意这里必须只喂训练集：
        - train 用来估计数据分布；
        - val / test 只能复用 train 的统计量；
        - 如果把 val / test 也混进来，会造成数据泄漏。
        """
        self.input_cols = list(input_cols)
        self.target_cols = list(target_cols)
        required_cols = list(dict.fromkeys([*self.input_cols, *self.target_cols]))
        self._x_scaler = SklearnStandardScaler()
        self._y_scaler = SklearnStandardScaler()

        for file in files:
            with Path(file).open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                _validate_columns(reader.fieldnames, required_cols, file)
                x_rows: list[list[float]] = []
                y_rows: list[list[float]] = []
                for row_idx, row in enumerate(reader, start=2):
                    # 同一行数据会被拆成两份：
                    # x_rows 只保留输入列，y_rows 只保留目标列。
                    x_rows.append(_read_float_values(row, self.input_cols, file, row_idx))
                    y_rows.append(_read_float_values(row, self.target_cols, file, row_idx))
            if x_rows:
                # partial_fit 允许我们按“文件块”累计统计量，
                # 不需要一次把所有 CSV 全读进内存。
                self._x_scaler.partial_fit(np.asarray(x_rows, dtype=np.float64))
                self._y_scaler.partial_fit(np.asarray(y_rows, dtype=np.float64))

        if self._x_scaler is None or self._y_scaler is None or not hasattr(self._x_scaler, "mean_"):
            raise ValueError("Cannot fit scaler: no valid rows found in train files.")

        self.x_mean = self._x_scaler.mean_.tolist()
        self.x_std = self._x_scaler.scale_.tolist()
        self.y_mean = self._y_scaler.mean_.tolist()
        self.y_std = self._y_scaler.scale_.tolist()
        return self

    def transform_x(self, array: Any) -> Any:
        """标准化输入特征 x。

        常见输入形状：
        - [feature]
        - [time, feature]
        - [batch, time, feature]

        约定始终是“最后一维 = 特征维”。
        """
        self._check_fitted()
        return _apply_last_dim(array, self.x_mean, self.x_std, inverse=False, scaler=self._x_scaler)

    def transform_y(self, array: Any) -> Any:
        """标准化目标 y，规则与 transform_x 相同，但使用 y 的统计量。

        之所以单独做一个方法，而不是复用 transform_x，
        是因为 y 的均值和标准差通常与 x 不同。
        """
        self._check_fitted()
        return _apply_last_dim(array, self.y_mean, self.y_std, inverse=False, scaler=self._y_scaler)

    def inverse_y(self, array: Any) -> Any:
        """把标准化后的 y 还原回原始物理量。

        这个方法只处理 y，不处理 x。
        原因是训练/推理结束后，真正需要恢复到物理空间做评估或可视化的，
        通常是模型预测出来的目标量。
        """
        self._check_fitted()
        return _apply_last_dim(array, self.y_mean, self.y_std, inverse=True, scaler=self._y_scaler)

    def inverse_y_tensor(self, tensor: Any) -> Any:
        """专门处理 torch.Tensor 版的反标准化。

        这样做可以避免把 Tensor 先转成 Python 列表，
        保持张量运算效率，也便于后续继续参与模型计算。
        """
        self._check_fitted()
        torch = _try_import_torch()
        if torch is not None and isinstance(tensor, torch.Tensor):
            # 构造可广播（broadcast）的 mean / std 形状，
            # 使其能够自动匹配 batch、time 等前置维度。
            shape = [1] * (tensor.dim() - 1) + [len(self.target_cols)]
            mean = tensor.new_tensor(self.y_mean).view(*shape)
            std = tensor.new_tensor(self.y_std).view(*shape)
            return tensor * std + mean
        return self.inverse_y(tensor)

    def save(self, path: str | Path) -> None:
        """把 scaler 参数保存到 JSON，方便下次直接加载。

        保存的是项目需要的最小信息：
        - 哪些列属于 input / target
        - x / y 各自的 mean/std

        没有直接 pickle 整个 sklearn 对象，是为了：
        - 文件更透明，能直接查看；
        - 跨环境更稳，不依赖 pickle 兼容性。
        """
        self._check_fitted()
        save_json(
            {
                "input_cols": self.input_cols,
                "target_cols": self.target_cols,
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "StandardScaler":
        """从 JSON 文件恢复一个已经拟合好的 scaler。

        这常用于：
        - 训练后保存 scaler；
        - 推理或评估时再次加载同一套标准化参数。
        """
        data = load_json(path)
        return cls(
            input_cols=data["input_cols"],
            target_cols=data["target_cols"],
            x_mean=data["x_mean"],
            x_std=data["x_std"],
            y_mean=data["y_mean"],
            y_std=data["y_std"],
        )

    def _check_fitted(self) -> None:
        """确保 scaler 已经先执行过 fit。"""
        if (
            not self.x_mean
            or not self.x_std
            or not self.y_mean
            or not self.y_std
            or self._x_scaler is None
            or self._y_scaler is None
        ):
            raise RuntimeError("StandardScaler is not fitted.")


def _validate_columns(
    fieldnames: Sequence[str] | None,
    required_cols: Sequence[str],
    file: str | Path,
) -> None:
    """检查 CSV 表头是否满足要求。

    这里提前失败比后面静默出错更好，
    否则你可能训练了很久才发现列对不上。
    """
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
    """从一行 CSV 中读取指定列，并转成 float。

    如果某一列不是合法数字，会抛出带文件名、行号、列名的报错，
    方便你定位脏数据。
    """
    values: list[float] = []
    for col in cols:
        raw = row.get(col, "")
        try:
            values.append(float(raw))
        except ValueError as exc:
            raise ValueError(f"Bad numeric value in {file}:{row_idx}, column {col}: {raw!r}") from exc
    return values


def _apply_last_dim(
    array: Any,
    mean: Sequence[float],
    std: Sequence[float],
    inverse: bool,
    scaler: SklearnStandardScaler | None,
) -> Any:
    """沿着最后一个维度做标准化 / 反标准化。

    为什么是“最后一个维度”：
    - 常见数据形状是 [batch, time, feature]
    - feature 维通常放在最后
    - 所以只要沿最后一维处理，就能兼容向量、序列、批量序列
    """
    torch = _try_import_torch()
    if torch is not None and isinstance(array, torch.Tensor):
        # 构造成可广播形状，例如：
        # 原张量 [B, T, C] -> mean/std 变成 [1, 1, C]
        shape = [1] * (array.dim() - 1) + [len(mean)]
        mean_tensor = array.new_tensor(mean).view(*shape)
        std_tensor = array.new_tensor(std).view(*shape)
        return array * std_tensor + mean_tensor if inverse else (array - mean_tensor) / std_tensor

    if isinstance(array, np.ndarray):
        if array.shape and array.shape[-1] != len(mean):
            raise ValueError(f"Expected last dimension {len(mean)}, got {array.shape[-1]}")
        if scaler is None:
            raise RuntimeError("StandardScaler is not fitted.")
        original_shape = array.shape
        matrix = np.asarray(array, dtype=np.float64).reshape(-1, len(mean))
        transformed = scaler.inverse_transform(matrix) if inverse else scaler.transform(matrix)
        return transformed.reshape(original_shape).astype(np.float32, copy=False)

    if _is_vector(array):
        if len(array) != len(mean):
            raise ValueError(f"Expected last dimension {len(mean)}, got {len(array)}")
        # sklearn 的 transform 期望二维矩阵 [N, C]，
        # 所以单个向量要先包成一行，再在最后拆回来。
        transformed = _apply_sklearn_rows([array], scaler=scaler, inverse=inverse)
        return transformed[0]

    # 如果当前不是“特征向量”，就继续向内递归，
    # 直到找到最后一层 [feature] 为止。
    return [_apply_last_dim(item, mean, std, inverse, scaler) for item in array]


def _is_vector(value: Any) -> bool:
    """判断一个对象是否可以视为“一维特征向量”。

    例如：
    - [1.0, 2.0, 3.0] 是向量；
    - [[1.0, 2.0], [3.0, 4.0]] 不是向量，而是二维结构。
    """
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and (
        not value or not isinstance(value[0], Sequence)
    )


def _try_import_torch() -> Any:
    """尝试导入 PyTorch；未安装则返回 None。

    这样做可以让同一套标准化逻辑同时兼容：
    - 纯 Python / numpy 数据流程；
    - 依赖 torch.Tensor 的训练与推理流程。
    """
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def _restore_sklearn_scaler(
    mean: Sequence[float],
    std: Sequence[float],
) -> SklearnStandardScaler | None:
    """根据保存下来的 mean/std 重新拼出一个 sklearn scaler。

    这让项目在 load JSON 之后，仍然可以继续复用
    sklearn 的 transform / inverse_transform 接口。
    """
    if not mean or not std:
        return None

    scaler = SklearnStandardScaler()
    mean_array = np.asarray(mean, dtype=np.float64)
    scale_array = np.asarray(std, dtype=np.float64)
    scaler.mean_ = mean_array
    scaler.scale_ = scale_array
    scaler.var_ = np.square(scale_array)
    scaler.n_features_in_ = len(mean_array)
    scaler.n_samples_seen_ = 1
    return scaler


def _apply_sklearn_rows(
    rows: Sequence[Sequence[float]],
    scaler: SklearnStandardScaler | None,
    inverse: bool,
) -> list[list[float]]:
    """把二维行向量送进 sklearn 做变换。

    sklearn 的 `transform()` / `inverse_transform()` 都假设输入是二维矩阵：
    - 行表示样本数 N
    - 列表示特征数 C

    所以哪怕你只有一个向量，也要先包装成 `[[...]]` 再送进去。
    """
    if scaler is None:
        raise RuntimeError("StandardScaler is not fitted.")

    matrix = np.asarray(rows, dtype=np.float64)
    transformed = scaler.inverse_transform(matrix) if inverse else scaler.transform(matrix)
    return transformed.tolist()

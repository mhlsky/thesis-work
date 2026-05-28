from __future__ import annotations

"""训练/评估时使用的硬件感知默认设置。"""

import ctypes
import math
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch


def _is_auto_like(value: Any) -> bool:
    """判断一个配置值是否表示“自动推断”."""
    return value is None or (isinstance(value, str) and value.strip().lower() == "auto")


def _read_env_text(name: str) -> str | None:
    """读取环境变量文本；空字符串按未设置处理。"""
    value = os.environ.get(name)
    if value is None:
        return None
    text = value.strip()
    return text or None


def _read_env_int(name: str) -> int | None:
    """读取整型环境变量。"""
    text = _read_env_text(name)
    if text is None:
        return None
    return int(text)


def _read_env_float(name: str) -> float | None:
    """读取浮点型环境变量。"""
    text = _read_env_text(name)
    if text is None:
        return None
    return float(text)


def _read_env_bool(name: str) -> bool | None:
    """读取布尔环境变量。"""
    text = _read_env_text(name)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Unsupported boolean env value: {name}={text}")


def _coerce_positive_int(value: Any) -> int:
    """把配置里的正整数解析成 int。"""
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"Expected a positive integer, got: {value!r}")
    return resolved


def _coerce_non_negative_int(value: Any) -> int:
    """把配置里的非负整数解析成 int。"""
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"Expected a non-negative integer, got: {value!r}")
    return resolved


def _coerce_bool(value: Any) -> bool:
    """把配置中的 true/false/1/0/yes/no 安全解析成 bool。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _round_down_to_multiple(value: float, multiple: int = 8, minimum: int = 8) -> int:
    """把数值向下取整到指定倍数，避免 batch size 太零碎。"""
    if value <= minimum:
        return minimum
    rounded = int(math.floor(value / multiple) * multiple)
    return max(rounded, minimum)


def detect_available_cpu_cores() -> int:
    """尽量获取当前进程真正可用的 CPU 核数。"""
    hinted = _read_env_int("SHIP_MOTION_CPU_CORES")
    if hinted is not None:
        return max(hinted, 1)
    try:
        return max(len(os.sched_getaffinity(0)), 1)
    except (AttributeError, OSError):
        return max(os.cpu_count() or 1, 1)


def detect_system_memory_gb() -> float | None:
    """估计当前机器的物理内存大小（GB）。"""
    hinted = _read_env_float("SHIP_MOTION_SYSTEM_MEMORY_GB")
    if hinted is not None:
        return hinted

    if os.name == "nt":
        class _MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("sullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        memory_status = _MemoryStatus()
        memory_status.dwLength = ctypes.sizeof(_MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
            return float(memory_status.ullTotalPhys) / (1024**3)
        return None

    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
            return float(page_size * phys_pages) / (1024**3)
        except (OSError, ValueError):
            return None
    return None


def detect_gpu_memory_gb(device: torch.device) -> float | None:
    """估计当前 CUDA 设备显存大小（GB）。"""
    hinted = _read_env_float("SHIP_MOTION_GPU_MEMORY_GB")
    if hinted is not None:
        return hinted
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    properties = torch.cuda.get_device_properties(device)
    return float(properties.total_memory) / (1024**3)


@dataclass(slots=True)
class HardwareSnapshot:
    """当前训练进程看到的关键硬件信息。"""

    profile: str
    device_type: str
    gpu_count: int
    gpu_memory_gb: float | None
    cpu_cores: int
    system_memory_gb: float | None
    bf16_supported: bool


def resolve_hardware_snapshot(device: torch.device, hardware_cfg: dict[str, Any] | None = None) -> HardwareSnapshot:
    """汇总环境变量、配置和运行时探测到的硬件信息。"""
    hardware_cfg = hardware_cfg or {}
    explicit_profile = _read_env_text("SHIP_MOTION_HW_PROFILE")
    if explicit_profile is None:
        profile_value = hardware_cfg.get("profile")
        explicit_profile = None if _is_auto_like(profile_value) else str(profile_value).strip()

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_memory_gb = detect_gpu_memory_gb(device)

    cpu_hint = None if _is_auto_like(hardware_cfg.get("cpu_cores")) else int(hardware_cfg["cpu_cores"])
    cpu_cores = cpu_hint if cpu_hint is not None else detect_available_cpu_cores()

    system_memory_hint = (
        None if _is_auto_like(hardware_cfg.get("system_memory_gb")) else float(hardware_cfg["system_memory_gb"])
    )
    system_memory_gb = system_memory_hint if system_memory_hint is not None else detect_system_memory_gb()

    if explicit_profile is not None:
        profile = explicit_profile
    elif device.type != "cuda":
        profile = "cpu_only"
    elif gpu_count == 1 and (gpu_memory_gb or 0.0) >= 38.0 and cpu_cores >= 10 and (system_memory_gb or 0.0) >= 28.0:
        profile = "single_gpu_40g_12cpu_32g"
    elif gpu_count == 1 and (gpu_memory_gb or 0.0) >= 22.0:
        profile = "single_gpu_midmem"
    else:
        profile = "single_gpu_cuda"

    bf16_supported = bool(
        device.type == "cuda"
        and torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )

    return HardwareSnapshot(
        profile=profile,
        device_type=device.type,
        gpu_count=gpu_count,
        gpu_memory_gb=gpu_memory_gb,
        cpu_cores=cpu_cores,
        system_memory_gb=system_memory_gb,
        bf16_supported=bf16_supported,
    )


def _default_batch_size(model_name: str, seq_len: int, pred_len: int, snapshot: HardwareSnapshot) -> int:
    """为当前模型族和硬件资源估计一个默认 batch size。"""
    if snapshot.device_type != "cuda":
        return 32

    base_by_model = {
        "persistence": 2048,
        "lstm": 512,
        "gru": 512,
        "tcn": 512,
        "transformer": 256,
        "lite_xlstm": 512,
        "ccg_xlstm": 512,
        "vmd_ccg_xlstm": 384,
    }
    base_value = float(base_by_model.get(model_name, 256))
    seq_scale = max(min(128.0 / max(float(seq_len), 1.0), 2.0), 0.25)
    pred_scale = max(min((10.0 / max(float(pred_len), 1.0)) ** 0.5, 1.5), 0.5)
    memory_scale = max(min((snapshot.gpu_memory_gb or 8.0) / 40.0, 2.0), 0.25)
    cpu_scale = 0.85 if snapshot.cpu_cores < 8 else 1.0
    ram_scale = 0.85 if (snapshot.system_memory_gb or 0.0) < 24.0 else 1.0
    estimated = base_value * seq_scale * pred_scale * memory_scale * cpu_scale * ram_scale
    return _round_down_to_multiple(estimated, multiple=8, minimum=8)


def _default_num_workers(snapshot: HardwareSnapshot) -> int:
    """根据 CPU / 内存资源给出较稳妥的 DataLoader worker 数。"""
    if snapshot.cpu_cores <= 2:
        return 0
    if snapshot.cpu_cores <= 4:
        return 2
    workers = min(8, max(4, snapshot.cpu_cores // 2))
    if (snapshot.system_memory_gb or 0.0) < 24.0:
        workers = min(workers, 4)
    return workers


def resolve_precision_setting(train_cfg: dict[str, Any], snapshot: HardwareSnapshot) -> str:
    """解析精度设置；auto 时优先用 bf16，否则回退 fp32。"""
    env_precision = _read_env_text("SHIP_MOTION_PRECISION")
    raw_precision = env_precision if env_precision is not None else train_cfg.get("precision", "auto")
    precision = "auto" if _is_auto_like(raw_precision) else str(raw_precision).strip().lower()
    if precision == "auto":
        if snapshot.device_type == "cuda" and snapshot.bf16_supported:
            return "bf16"
        return "fp32"
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError(f"Unsupported precision: {precision!r}. Expected one of fp32/bf16/fp16/auto.")
    if precision in {"bf16", "fp16"} and snapshot.device_type != "cuda":
        return "fp32"
    if precision == "bf16" and not snapshot.bf16_supported:
        return "fp32"
    return precision


def apply_hardware_aware_training_defaults(config: dict[str, Any], device: torch.device) -> HardwareSnapshot:
    """把训练配置里的 auto 字段解析成适合当前硬件的明确值。"""
    data_cfg = config.setdefault("data", {})
    train_cfg = config.setdefault("train", {})
    hardware_cfg = config.setdefault("hardware", {})
    snapshot = resolve_hardware_snapshot(device=device, hardware_cfg=hardware_cfg)

    model_name = str(config.get("model", {}).get("name", "")).strip().lower()
    seq_len = int(data_cfg.get("seq_len", 128))
    pred_len = int(data_cfg.get("pred_len", 10))

    default_batch_size = _default_batch_size(
        model_name=model_name,
        seq_len=seq_len,
        pred_len=pred_len,
        snapshot=snapshot,
    )
    batch_env = _read_env_int("SHIP_MOTION_BATCH_SIZE")
    raw_batch_size = batch_env if batch_env is not None else train_cfg.get("batch_size", "auto")
    train_cfg["batch_size"] = (
        default_batch_size if _is_auto_like(raw_batch_size) else _coerce_positive_int(raw_batch_size)
    )

    default_num_workers = _default_num_workers(snapshot)
    workers_env = _read_env_int("SHIP_MOTION_NUM_WORKERS")
    raw_num_workers = workers_env if workers_env is not None else train_cfg.get("num_workers", "auto")
    train_cfg["num_workers"] = (
        default_num_workers if _is_auto_like(raw_num_workers) else _coerce_non_negative_int(raw_num_workers)
    )

    default_prefetch_factor = None if train_cfg["num_workers"] <= 0 else (4 if train_cfg["num_workers"] >= 4 else 2)
    prefetch_env = _read_env_int("SHIP_MOTION_PREFETCH_FACTOR")
    raw_prefetch_factor = prefetch_env if prefetch_env is not None else train_cfg.get("prefetch_factor", "auto")
    if train_cfg["num_workers"] <= 0:
        train_cfg["prefetch_factor"] = None
    else:
        train_cfg["prefetch_factor"] = (
            default_prefetch_factor
            if _is_auto_like(raw_prefetch_factor)
            else _coerce_positive_int(raw_prefetch_factor)
        )

    pin_memory_env = _read_env_bool("SHIP_MOTION_PIN_MEMORY")
    raw_pin_memory = pin_memory_env if pin_memory_env is not None else train_cfg.get("pin_memory", "auto")
    train_cfg["pin_memory"] = snapshot.device_type == "cuda" if _is_auto_like(raw_pin_memory) else _coerce_bool(raw_pin_memory)

    persistent_env = _read_env_bool("SHIP_MOTION_PERSISTENT_WORKERS")
    raw_persistent = (
        persistent_env if persistent_env is not None else train_cfg.get("persistent_workers", "auto")
    )
    auto_persistent = train_cfg["num_workers"] > 0
    train_cfg["persistent_workers"] = auto_persistent if _is_auto_like(raw_persistent) else _coerce_bool(raw_persistent)
    if train_cfg["num_workers"] <= 0:
        train_cfg["persistent_workers"] = False

    non_blocking_env = _read_env_bool("SHIP_MOTION_NON_BLOCKING")
    raw_non_blocking = non_blocking_env if non_blocking_env is not None else train_cfg.get("non_blocking", "auto")
    auto_non_blocking = snapshot.device_type == "cuda" and bool(train_cfg["pin_memory"])
    train_cfg["non_blocking"] = auto_non_blocking if _is_auto_like(raw_non_blocking) else _coerce_bool(raw_non_blocking)

    train_cfg["precision"] = resolve_precision_setting(train_cfg=train_cfg, snapshot=snapshot)

    hardware_cfg["resolved_profile"] = snapshot.profile
    hardware_cfg["detected_device_type"] = snapshot.device_type
    hardware_cfg["detected_gpu_count"] = snapshot.gpu_count
    hardware_cfg["detected_cpu_cores"] = snapshot.cpu_cores
    if snapshot.gpu_memory_gb is not None:
        hardware_cfg["detected_gpu_memory_gb"] = round(snapshot.gpu_memory_gb, 2)
    if snapshot.system_memory_gb is not None:
        hardware_cfg["detected_system_memory_gb"] = round(snapshot.system_memory_gb, 2)
    return snapshot


def autocast_context(device: torch.device, precision: str):
    """根据精度设置创建 autocast 上下文；不需要时返回空上下文。"""
    normalized = str(precision).strip().lower()
    if device.type != "cuda":
        return nullcontext()
    if normalized == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def build_grad_scaler(device: torch.device, precision: str) -> torch.cuda.amp.GradScaler | None:
    """仅在显式 fp16 训练时启用 GradScaler。"""
    if device.type != "cuda" or str(precision).strip().lower() != "fp16":
        return None
    return torch.cuda.amp.GradScaler(enabled=True)

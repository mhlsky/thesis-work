from __future__ import annotations

"""实验5：对既有 checkpoint 进行模型复杂度和前向推理性能测试。

本模块不会训练模型，也不会读取数据集重新计算 OOD 指标。它只做三件事：
1. 从已有 ``config.yaml`` 和 ``best.pt`` 恢复模型；
2. 构造与正式任务一致形状的单样本输入；
3. 统计参数量、权重体积、前向延迟分位数、吞吐量和 GPU 峰值显存。

这样可以把性能测试与精度实验分开：精度继续引用原始实验的 metrics_ood_test.json，
而本模块提供可复现的部署开销数据。
"""

import argparse
import copy
import csv
import json
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from ship_motion.evaluate import load_model_from_config
from ship_motion.utils import ensure_dir, load_json, load_yaml, save_json


MIB = 1024 * 1024
CSV_FIELDS = [
    "profile_name",
    "device",
    "precision",
    "model_id",
    "model_label",
    "family",
    "source_result",
    "run_name",
    "seed",
    "ood_rmse_reference",
    "ood_reference_protocol",
    "input_shape",
    "output_shape",
    "output_type",
    "trainable_params",
    "total_params",
    "params_m",
    "state_dict_tensor_size_mib",
    "checkpoint_size_mib",
    "latency_mean_ms",
    "latency_std_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_min_ms",
    "latency_max_ms",
    "throughput_samples_per_sec",
    "gpu_peak_memory_allocated_mib",
    "cpu_threads",
    "warmup_runs",
    "measure_runs",
]


def _format_duration(seconds: float) -> str:
    """把秒数格式化成便于远程观察的简短时长。"""
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _log(message: str) -> None:
    """统一使用 flush 输出，保证远程运行时能及时看到进度。"""
    print(message, flush=True)


def _resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    """把配置中的相对路径统一解析到项目根目录。"""
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _parse_csv_names(value: str | None) -> set[str] | None:
    """解析命令行中逗号分隔的模型或 profile 名称。"""
    if value is None or not value.strip():
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _quantile(samples: Sequence[float], percentile: float) -> float:
    """返回线性插值分位数，避免额外引入 NumPy 依赖。"""
    if not samples:
        raise ValueError("Cannot calculate a quantile from an empty sample list.")
    ordered = sorted(samples)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def _resolve_device(device_name: str) -> torch.device:
    """解析性能测试 profile 的设备，并对不可用 CUDA 给出明确报错。"""
    normalized = str(device_name).strip().lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "The requested CUDA benchmark profile is unavailable. "
                "Run only cpu_fp32 locally, or restore the cloud GPU runtime before the formal CUDA benchmark."
            )
        return torch.device(normalized)
    raise ValueError(f"Unsupported benchmark device: {device_name!r}. Expected cpu or cuda[:index].")


def _autocast_context(device: torch.device, precision: str):
    """只在 CUDA 的 bf16/fp16 profile 下启用 autocast。"""
    normalized = str(precision).strip().lower()
    if device.type != "cuda" or normalized == "fp32":
        return nullcontext()
    if normalized == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if normalized == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError(f"Unsupported benchmark precision: {precision!r}. Expected fp32/bf16/fp16.")


def _synchronize(device: torch.device) -> None:
    """GPU 为异步执行；每次单请求计时需要同步才有真实延迟。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _extract_output_shape(output: Any) -> tuple[str, list[int] | None]:
    """统一记录 Tensor 或 VMD 模型字典输出的主预测形状。"""
    if isinstance(output, torch.Tensor):
        return "tensor", list(output.shape)
    if isinstance(output, Mapping):
        y_hat = output.get("y_hat")
        if isinstance(y_hat, torch.Tensor):
            return "mapping[y_hat]", list(y_hat.shape)
        return "mapping", None
    return type(output).__name__, None


def _state_dict_tensor_size_mib(model: torch.nn.Module) -> float:
    """计算 state_dict 中所有 tensor 的原始体积，不受 checkpoint 序列化格式影响。"""
    bytes_total = sum(value.numel() * value.element_size() for value in model.state_dict().values() if torch.is_tensor(value))
    return float(bytes_total / MIB)


def _model_parameter_stats(model: torch.nn.Module) -> tuple[int, int, float, float]:
    """返回可训练参数量、总参数量、百万参数量与 state_dict 原始体积。"""
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return trainable_params, total_params, float(total_params / 1_000_000), _state_dict_tensor_size_mib(model)


def _build_input(config: dict[str, Any], batch_size: int, input_seed: int, device: torch.device) -> torch.Tensor:
    """按 checkpoint 对应数据配置生成固定形状的代表性输入，不读取任何数据文件。"""
    data_cfg = config.get("data", {})
    seq_len = int(data_cfg["seq_len"])
    input_dim = len(data_cfg.get("input_cols", []))
    if input_dim <= 0:
        raise ValueError("Runtime config must define a non-empty data.input_cols for inference benchmarking.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(input_seed))
    input_cpu = torch.randn((batch_size, seq_len, input_dim), generator=generator, dtype=torch.float32)
    return input_cpu.to(device)


def _run_forward(model: torch.nn.Module, example_input: torch.Tensor, device: torch.device, precision: str) -> Any:
    """执行一次纯前向；Physics loss、VMD cache 和数据预处理均不属于本计时范围。"""
    with torch.inference_mode(), _autocast_context(device, precision):
        return model(example_input)


def _measure_latency(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    device: torch.device,
    precision: str,
    warmup_runs: int,
    measure_runs: int,
) -> tuple[list[float], float | None]:
    """预热后逐次测量前向延迟，并在 CUDA 上记录峰值分配显存。"""
    if warmup_runs < 0 or measure_runs <= 0:
        raise ValueError("warmup_runs must be non-negative and measure_runs must be positive.")

    for _ in range(warmup_runs):
        _run_forward(model, example_input, device, precision)
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    for _ in range(measure_runs):
        _synchronize(device)
        started_at = time.perf_counter_ns()
        _run_forward(model, example_input, device, precision)
        _synchronize(device)
        latencies_ms.append((time.perf_counter_ns() - started_at) / 1_000_000.0)

    peak_memory_mib = None
    if device.type == "cuda":
        peak_memory_mib = float(torch.cuda.max_memory_allocated(device) / MIB)
    return latencies_ms, peak_memory_mib


def _read_ood_rmse(metrics_path: Path) -> float | None:
    """读取既有 OOD 指标，仅供同表展示，绝不触发重新评估。"""
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    value = metrics.get("rmse_mean") if isinstance(metrics, dict) else None
    return None if value is None else float(value)


def _profile_metadata(device: torch.device) -> dict[str, Any]:
    """记录影响延迟结果的运行环境，便于论文和后续复测解释。"""
    metadata: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        metadata["gpu_total_memory_mib"] = float(torch.cuda.get_device_properties(device).total_memory / MIB)
    return metadata


def _select_items(items: Iterable[dict[str, Any]], requested_names: set[str] | None, key: str) -> list[dict[str, Any]]:
    """按名称筛选配置条目，并避免静默忽略拼写错误。"""
    materialized = list(items)
    if requested_names is None:
        return materialized
    available = {str(item[key]) for item in materialized}
    missing = sorted(requested_names - available)
    if missing:
        raise ValueError(f"Unknown {key} value(s): {', '.join(missing)}. Available: {', '.join(sorted(available))}")
    return [item for item in materialized if str(item[key]) in requested_names]


def benchmark_from_config(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    profile_names: set[str] | None = None,
    model_ids: set[str] | None = None,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """运行配置驱动的性能测试并写出 CSV/JSON。"""
    source_path = Path(config_path).resolve()
    repo_root = source_path.parent.parent
    spec = load_yaml(source_path)
    benchmark_cfg = copy.deepcopy(spec.get("benchmark", {}))
    if not isinstance(benchmark_cfg, dict):
        raise ValueError("experiment5 config must define a mapping named benchmark.")

    configured_profiles = benchmark_cfg.get("profiles", [])
    configured_models = spec.get("models", [])
    if not isinstance(configured_profiles, list) or not configured_profiles:
        raise ValueError("benchmark.profiles must be a non-empty list.")
    if not isinstance(configured_models, list) or not configured_models:
        raise ValueError("models must be a non-empty list.")

    overall_started_at = time.perf_counter()
    if smoke:
        smoke_cfg = benchmark_cfg.get("smoke", {})
        if not isinstance(smoke_cfg, dict):
            raise ValueError("benchmark.smoke must be a mapping.")
        profile_names = set(smoke_cfg.get("profiles", [])) if profile_names is None else profile_names
        model_ids = set(smoke_cfg.get("model_ids", [])) if model_ids is None else model_ids
        warmup_runs = int(smoke_cfg.get("warmup_runs", 3))
        measure_runs = int(smoke_cfg.get("measure_runs", 10))
    else:
        warmup_runs = int(benchmark_cfg.get("warmup_runs", 100))
        measure_runs = int(benchmark_cfg.get("measure_runs", 1000))

    profiles = _select_items(configured_profiles, profile_names, key="name")
    models = _select_items(configured_models, model_ids, key="id")
    target_dir = _resolve_repo_path(repo_root, output_dir or benchmark_cfg["output_dir"])
    ensure_dir(target_dir)

    batch_size = int(benchmark_cfg.get("batch_size", 1))
    input_seed = int(benchmark_cfg.get("input_seed", 2026))
    rows: list[dict[str, Any]] = []
    metadata_profiles: list[dict[str, Any]] = []
    total_jobs = len(profiles) * len(models)
    completed_jobs = 0

    _log(
        "[Benchmark] start "
        f"smoke={smoke} batch_size={batch_size} warmup_runs={warmup_runs} measure_runs={measure_runs} "
        f"profiles={len(profiles)} models={len(models)} output_dir={target_dir}"
    )

    for profile in profiles:
        profile_name = str(profile["name"])
        device = _resolve_device(str(profile["device"]))
        precision = str(profile.get("precision", "fp32")).lower()
        cpu_threads = profile.get("cpu_threads")
        original_threads = torch.get_num_threads()
        if cpu_threads is not None:
            torch.set_num_threads(int(cpu_threads))

        profile_started_at = time.perf_counter()
        _log(
            f"[Benchmark] profile start name={profile_name} device={device} "
            f"precision={precision} cpu_threads={cpu_threads if cpu_threads is not None else '<unchanged>'}"
        )
        try:
            metadata_profiles.append(
                {
                    "name": profile_name,
                    "precision": precision,
                    "cpu_threads": cpu_threads,
                    "environment": _profile_metadata(device),
                }
            )
            for model_spec in models:
                model_config_path = _resolve_repo_path(repo_root, model_spec["config_path"])
                checkpoint_path = _resolve_repo_path(repo_root, model_spec["checkpoint_path"])
                metrics_path = _resolve_repo_path(repo_root, model_spec["metrics_path"])
                if not model_config_path.is_file():
                    raise FileNotFoundError(f"Runtime config not found: {model_config_path}")
                if not checkpoint_path.is_file():
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

                job_started_at = time.perf_counter()
                completed_jobs += 1
                _log(
                    f"[Benchmark] job {completed_jobs}/{total_jobs} load model={model_spec['id']} "
                    f"run={model_spec['run_name']}"
                )
                runtime_config = load_yaml(model_config_path)
                model = load_model_from_config(runtime_config, checkpoint_path=checkpoint_path, device=device)
                model.eval()
                example_input = _build_input(runtime_config, batch_size=batch_size, input_seed=input_seed, device=device)
                sample_output = _run_forward(model, example_input, device, precision)
                _synchronize(device)
                output_type, output_shape = _extract_output_shape(sample_output)
                trainable_params, total_params, params_m, state_dict_size_mib = _model_parameter_stats(model)
                latencies_ms, peak_memory_mib = _measure_latency(
                    model=model,
                    example_input=example_input,
                    device=device,
                    precision=precision,
                    warmup_runs=warmup_runs,
                    measure_runs=measure_runs,
                )
                mean_latency_ms = float(statistics.fmean(latencies_ms))
                row = {
                    "profile_name": profile_name,
                    "device": str(device),
                    "precision": precision,
                    "model_id": str(model_spec["id"]),
                    "model_label": str(model_spec["label"]),
                    "family": str(model_spec["family"]),
                    "source_result": str(model_spec["source_result"]),
                    "run_name": str(model_spec["run_name"]),
                    "seed": int(model_spec["seed"]),
                    "ood_rmse_reference": _read_ood_rmse(metrics_path),
                    "ood_reference_protocol": str(model_spec.get("ood_reference_protocol", "existing result")),
                    "input_shape": json.dumps(list(example_input.shape)),
                    "output_shape": json.dumps(output_shape) if output_shape is not None else "",
                    "output_type": output_type,
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "params_m": params_m,
                    "state_dict_tensor_size_mib": state_dict_size_mib,
                    "checkpoint_size_mib": float(checkpoint_path.stat().st_size / MIB),
                    "latency_mean_ms": mean_latency_ms,
                    "latency_std_ms": float(statistics.pstdev(latencies_ms)),
                    "latency_p50_ms": _quantile(latencies_ms, 0.50),
                    "latency_p95_ms": _quantile(latencies_ms, 0.95),
                    "latency_p99_ms": _quantile(latencies_ms, 0.99),
                    "latency_min_ms": float(min(latencies_ms)),
                    "latency_max_ms": float(max(latencies_ms)),
                    "throughput_samples_per_sec": float(1000.0 / mean_latency_ms),
                    "gpu_peak_memory_allocated_mib": peak_memory_mib,
                    "cpu_threads": cpu_threads,
                    "warmup_runs": warmup_runs,
                    "measure_runs": measure_runs,
                }
                rows.append(row)
                _log(
                    f"[Benchmark] job {completed_jobs}/{total_jobs} done profile={profile_name} "
                    f"model={row['model_id']} p95={row['latency_p95_ms']:.4f}ms "
                    f"mean={row['latency_mean_ms']:.4f}ms params={row['params_m']:.4f}M "
                    f"elapsed={_format_duration(time.perf_counter() - job_started_at)}"
                )
                del sample_output, example_input, model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        finally:
            torch.set_num_threads(original_threads)
            _log(
                f"[Benchmark] profile done name={profile_name} "
                f"elapsed={_format_duration(time.perf_counter() - profile_started_at)}"
            )

    csv_path = target_dir / "model_benchmark.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        {
            "source_config": str(source_path),
            "timing_scope": "loaded model forward only; excludes checkpoint loading, data preprocessing, VMD cache construction, and physics loss",
            "smoke": smoke,
            "batch_size": batch_size,
            "input_seed": input_seed,
            "profiles": metadata_profiles,
            "row_count": len(rows),
        },
        target_dir / "benchmark_metadata.json",
    )
    _log(
        f"[Benchmark] done rows={len(rows)} total_elapsed={_format_duration(time.perf_counter() - overall_started_at)} "
        f"csv={csv_path}"
    )
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Benchmark existing ship-motion model checkpoints without retraining.")
    parser.add_argument("--config", default="configs/experiment5_performance.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--profiles", default=None, help="Comma-separated profile names, for example cpu_fp32,cuda_fp32.")
    parser.add_argument("--models", default=None, help="Comma-separated model IDs, for example lstm,joint_final.")
    parser.add_argument("--smoke", action="store_true", help="Use the config's tiny CPU smoke profile and model subset.")
    args = parser.parse_args(argv)
    benchmark_from_config(
        config_path=args.config,
        output_dir=args.output_dir,
        profile_names=_parse_csv_names(args.profiles),
        model_ids=_parse_csv_names(args.models),
        smoke=bool(args.smoke),
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

"""为正式实验生成统一的 results/<result_name>/ 运行配置。"""

import argparse
from pathlib import Path
from typing import Any, Sequence

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIGS = [
    "configs/persistence.yaml",
    "configs/lstm.yaml",
    "configs/gru.yaml",
    "configs/transformer.yaml",
    "configs/lite_xlstm.yaml",
    "configs/ccg_xlstm.yaml",
    "configs/vmd_ccg_xlstm.yaml",
    "configs/vmd_ccg_phys_xlstm.yaml",
]


def prepare_formal_result_configs(
    result_name: str,
    configs: Sequence[str] | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> list[Path]:
    """把正式实验配置改写到 results/<result_name>/runtime_configs 下。"""
    root = Path(repo_root).resolve()
    selected_configs = list(configs or DEFAULT_CONFIGS)
    if not selected_configs:
        raise ValueError("At least one config path must be provided.")

    result_dir = root / "results" / result_name
    output_root = result_dir / "outputs"
    runtime_config_root = result_dir / "runtime_configs"
    output_root.mkdir(parents=True, exist_ok=True)
    runtime_config_root.mkdir(parents=True, exist_ok=True)

    generated_paths: list[Path] = []
    for config_text in selected_configs:
        source_path = (root / config_text).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Config not found: {source_path}")
        target_path = runtime_config_root / source_path.name
        config = load_yaml(source_path)
        rewritten = rewrite_formal_runtime_config(config, output_root=output_root)
        save_yaml(target_path, rewritten)
        generated_paths.append(target_path)
    return generated_paths


def rewrite_formal_runtime_config(config: dict[str, Any], output_root: Path) -> dict[str, Any]:
    """把正式实验输出统一改写到指定结果目录。"""
    rewritten = dict(config)
    rewritten["output_dir"] = to_repo_relative(output_root)

    vmd_cfg = rewritten.get("vmd")
    if isinstance(vmd_cfg, dict) and bool(vmd_cfg.get("enabled", False)):
        # 正式实验缓存也统一收到当前 result 目录里，避免多个实验互相覆盖。
        vmd_cfg = dict(vmd_cfg)
        vmd_cfg["cache_root"] = to_repo_relative(output_root / "cache" / "vmd" / vmd_cache_dir_name(vmd_cfg))
        rewritten["vmd"] = vmd_cfg
    return rewritten


def vmd_cache_dir_name(vmd_cfg: dict[str, Any]) -> str:
    """根据 VMD 参数生成缓存目录名。"""
    k = int(vmd_cfg.get("K", 3))
    alpha = float(vmd_cfg.get("alpha", 2000.0))
    alpha_str = f"{alpha:g}".replace(".", "_")
    return f"K{k}_alpha{alpha_str}"


def to_repo_relative(path: Path) -> str:
    """尽量把路径写成相对仓库根目录的形式，便于跨机器复现。"""
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {path}, got: {type(data)!r}")
    return data


def save_yaml(path: str | Path, data: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare runtime configs so a formal experiment writes everything under results/<result_name>/."
    )
    parser.add_argument("--result-name", required=True, help="Target result directory name under results/.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional config list. Defaults to the main Step 07 formal run set.",
    )
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    generated_paths = prepare_formal_result_configs(
        result_name=args.result_name,
        configs=args.configs,
        repo_root=args.repo_root,
    )
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail

# 实验4：Phys / VMD 精修与联合验证（Linux 云服务器版）。
#
# 设计目标：
# 1. 沿用 result3 的“运行时配置改写”方式，不直接改核心训练代码；
# 2. 先做单模块宽扫与诊断，再做少量联合验证；
# 3. 单 seed 主实验统一汇总到 configs/ablation_result4.yaml；
# 4. 多随机种子 robust 组单独运行，不默认混入主表；
# 5. 所有正式输出统一写到 results/result4/ 下。
#
# 默认运行 full（单 seed 主实验 + robust 多 seed）。
# 可选环境变量：
# - RUN_GROUP=full|all|baseline|phys_main|phys_diag|phys|vmd_main|vmd_diag|vmd|joint|robust
# - RUN_ONLY=cfg1,cfg2               只跑指定 runtime config（不带 .yaml），优先级最高
# - PYTHON_BIN=python                指定解释器，默认使用当前激活环境里的 python
# - SKIP_SUMMARY=1                   跳过默认主表汇总
# - ROBUST_SEEDS=7,42,2026,...       覆盖 robust 组默认种子
# - ROBUST_BASES=stem1,stem2,...     覆盖 robust 组默认基配置
#
# 用法：
#   conda activate your_env
#   bash scripts/run_result4_experiments.sh . result4
#   RUN_GROUP=all bash scripts/run_result4_experiments.sh . result4
#   RUN_GROUP=phys bash scripts/run_result4_experiments.sh . result4
#   RUN_GROUP=robust ROBUST_SEEDS=7,42,1234 bash scripts/run_result4_experiments.sh . result4

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result4}"
RUN_GROUP="${RUN_GROUP:-full}"
RUN_ONLY="${RUN_ONLY:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"
ROBUST_SEEDS="${ROBUST_SEEDS:-7,42,2026,3407,10007}"
ROBUST_BASES="${ROBUST_BASES:-e4_ccg_xlstm_base,e4_ccg_phys_s0010_r0050,e4_ccg_phys_s0010_r0000,e4_vmd_ccg_l005_lr5e4_mix,e4_vmd_ccg_l005_lr5e4_nomix,e4_joint_l005_lr5e4_nomix_r0050}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="$PROJECT_ROOT/results/$RESULT_NAME"
OUTPUT_ROOT="$RESULT_DIR/outputs"
LOG_ROOT="$RESULT_DIR/logs"
SUMMARY_ROOT="$OUTPUT_ROOT/summary"
RUNTIME_CONFIG_ROOT="$PROJECT_ROOT/runtime_configs_${RESULT_NAME}"
ABLATION_CONFIG="$PROJECT_ROOT/configs/ablation_result4.yaml"

mkdir -p "$RESULT_DIR" "$OUTPUT_ROOT" "$LOG_ROOT" "$SUMMARY_ROOT" "$RUNTIME_CONFIG_ROOT"
PIPELINE_START_TS=$(date +%s)

format_duration() {
  local total_seconds="${1:-0}"
  local hours=$((total_seconds / 3600))
  local minutes=$(((total_seconds % 3600) / 60))
  local seconds=$((total_seconds % 60))
  printf "%02d:%02d:%02d" "$hours" "$minutes" "$seconds"
}

print_banner() {
  local message="$1"
  echo "================================================================================"
  echo "$message"
  echo "================================================================================"
}

run_cmd() {
  local name="$1"
  local stage_label="$2"
  shift 2
  local task_start_ts
  task_start_ts=$(date +%s)
  print_banner "[Task Start] name=$name | stage=$stage_label | started_at=$(date '+%F %T')"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  local task_end_ts
  task_end_ts=$(date +%s)
  print_banner "[Task Done] name=$name | stage=$stage_label | elapsed=$(format_duration "$((task_end_ts - task_start_ts))") | log=$LOG_ROOT/${name}.log"
}

array_contains() {
  local needle="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

trim_csv_into_array() {
  local text="$1"
  local -n target_ref="$2"
  target_ref=()
  IFS=',' read -r -a _raw_items <<< "$text"
  local item
  for item in "${_raw_items[@]}"; do
    item="${item//[[:space:]]/}"
    if [[ -n "$item" ]]; then
      target_ref+=("$item")
    fi
  done
}

print_banner "[Run Start] Result4 experiment pipeline | project_root=$PROJECT_ROOT | result_name=$RESULT_NAME | run_group=$RUN_GROUP"

run_cmd "prepare_result4_runtime_configs" "prepare_configs" \
  "$PYTHON_BIN" - "$PROJECT_ROOT" "$RESULT_NAME" "$RUNTIME_CONFIG_ROOT" "$ROBUST_SEEDS" "$ROBUST_BASES" <<'PY'
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

project_root = Path(sys.argv[1]).resolve()
result_name = sys.argv[2]
runtime_config_root = Path(sys.argv[3]).resolve()
robust_seeds = [int(item.strip()) for item in sys.argv[4].split(",") if item.strip()]
robust_bases = [item.strip() for item in sys.argv[5].split(",") if item.strip()]

configs_root = project_root / "configs"
output_dir = f"results/{result_name}/outputs"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping yaml at {path}")
    return data


def save_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def with_common_output(cfg: dict, run_name: str) -> dict:
    rewritten = copy.deepcopy(cfg)
    rewritten["output_dir"] = output_dir
    rewritten["run_name"] = run_name
    train_cfg = rewritten.setdefault("train", {})
    train_cfg.setdefault("seed", 42)
    return rewritten


def build_physics_section(
    *,
    lambda_smooth: float,
    lambda_roll: float,
    warmup_epochs: int = 8,
    normalize_by_target_std: bool = True,
    roll_integration: str = "trapezoid",
) -> dict:
    return {
        "enabled": True,
        "lambda_smooth": float(lambda_smooth),
        "lambda_roll": float(lambda_roll),
        "dt": 1.0,
        "warmup_epochs": int(warmup_epochs),
        "smoothness_target_cols": ["p", "r", "phi"],
        "normalize_by_target_std": bool(normalize_by_target_std),
        "roll_integration": str(roll_integration),
        "p_col": "p",
        "phi_col": "phi",
    }


def build_vmd_section(
    *,
    lambda_vmd: float,
    warmup_epochs: int,
    K: int = 3,
    alpha: int = 2000,
) -> dict:
    return {
        "enabled": True,
        "K": int(K),
        "alpha": int(alpha),
        "tau": 0.0,
        "DC": False,
        "init": 1,
        "tol": 1.0e-7,
        "lambda_vmd": float(lambda_vmd),
        "warmup_epochs": int(warmup_epochs),
        "cache_root": f"results/{result_name}/outputs/cache/vmd/K{K}_alpha{alpha}",
    }


def build_phys_config(
    ccg_base: dict,
    run_name: str,
    *,
    lambda_smooth: float,
    lambda_roll: float,
    warmup_epochs: int = 8,
    normalize_by_target_std: bool = True,
    roll_integration: str = "trapezoid",
) -> dict:
    cfg = with_common_output(ccg_base, run_name)
    cfg["physics"] = build_physics_section(
        lambda_smooth=lambda_smooth,
        lambda_roll=lambda_roll,
        warmup_epochs=warmup_epochs,
        normalize_by_target_std=normalize_by_target_std,
        roll_integration=roll_integration,
    )
    return cfg


def build_vmd_config(
    vmd_base: dict,
    run_name: str,
    *,
    lambda_vmd: float = 0.05,
    lr: float = 1.0e-3,
    use_state_mixer: bool = True,
    state_mixer_hidden_dim: int = 16,
    state_mixer_dropout: float = 0.0,
    warmup_epochs: int = 5,
    K: int = 3,
    alpha: int = 2000,
) -> dict:
    cfg = with_common_output(vmd_base, run_name)
    cfg["vmd"] = build_vmd_section(
        lambda_vmd=lambda_vmd,
        warmup_epochs=warmup_epochs,
        K=K,
        alpha=alpha,
    )
    cfg.setdefault("train", {})["lr"] = float(lr)
    model_cfg = cfg.setdefault("model", {})
    model_cfg["use_state_mixer"] = bool(use_state_mixer)
    model_cfg["state_mixer_hidden_dim"] = int(state_mixer_hidden_dim)
    model_cfg["state_mixer_dropout"] = float(state_mixer_dropout)
    return cfg


def build_joint_config(
    vmd_base: dict,
    run_name: str,
    *,
    lambda_vmd: float = 0.05,
    lr: float = 5.0e-4,
    use_state_mixer: bool = False,
    state_mixer_hidden_dim: int = 16,
    state_mixer_dropout: float = 0.0,
    vmd_warmup_epochs: int = 5,
    lambda_smooth: float = 0.0,
    lambda_roll: float = 0.0,
    physics_warmup_epochs: int = 8,
    normalize_by_target_std: bool = True,
    roll_integration: str = "trapezoid",
) -> dict:
    cfg = build_vmd_config(
        vmd_base=vmd_base,
        run_name=run_name,
        lambda_vmd=lambda_vmd,
        lr=lr,
        use_state_mixer=use_state_mixer,
        state_mixer_hidden_dim=state_mixer_hidden_dim,
        state_mixer_dropout=state_mixer_dropout,
        warmup_epochs=vmd_warmup_epochs,
        K=3,
        alpha=2000,
    )
    cfg["physics"] = build_physics_section(
        lambda_smooth=lambda_smooth,
        lambda_roll=lambda_roll,
        warmup_epochs=physics_warmup_epochs,
        normalize_by_target_std=normalize_by_target_std,
        roll_integration=roll_integration,
    )
    return cfg


ccg_base = load_yaml(configs_root / "ccg_xlstm.yaml")
vmd_base = load_yaml(configs_root / "vmd_ccg_xlstm.yaml")

result4_configs: dict[str, dict] = {}

# ------------------------------------------------------------------
# 基线组
# ------------------------------------------------------------------
result4_configs["e4_ccg_xlstm_base"] = with_common_output(ccg_base, "e4_ccg_xlstm_base")

# ------------------------------------------------------------------
# Phys 系统扫描组
# ------------------------------------------------------------------
phys_specs = {
    "e4_ccg_phys_s0005_r0000": dict(lambda_smooth=0.0005, lambda_roll=0.0),
    "e4_ccg_phys_s0010_r0000": dict(lambda_smooth=0.0010, lambda_roll=0.0),
    "e4_ccg_phys_s0000_r0025": dict(lambda_smooth=0.0, lambda_roll=0.0025),
    "e4_ccg_phys_s0000_r0050": dict(lambda_smooth=0.0, lambda_roll=0.0050),
    "e4_ccg_phys_s0000_r0075": dict(lambda_smooth=0.0, lambda_roll=0.0075),
    "e4_ccg_phys_s0005_r0025": dict(lambda_smooth=0.0005, lambda_roll=0.0025),
    "e4_ccg_phys_s0005_r0050": dict(lambda_smooth=0.0005, lambda_roll=0.0050),
    "e4_ccg_phys_s0010_r0050": dict(lambda_smooth=0.0010, lambda_roll=0.0050),
    "e4_ccg_phys_s0010_r0075": dict(lambda_smooth=0.0010, lambda_roll=0.0075),
    "e4_ccg_phys_s0010_r0100": dict(lambda_smooth=0.0010, lambda_roll=0.0100),
}
for stem, kwargs in phys_specs.items():
    result4_configs[stem] = build_phys_config(ccg_base, stem, **kwargs)

result4_configs["e4_ccg_phys_s0010_r0050_normoff"] = build_phys_config(
    ccg_base,
    "e4_ccg_phys_s0010_r0050_normoff",
    lambda_smooth=0.0010,
    lambda_roll=0.0050,
    normalize_by_target_std=False,
)
result4_configs["e4_ccg_phys_s0010_r0050_euler"] = build_phys_config(
    ccg_base,
    "e4_ccg_phys_s0010_r0050_euler",
    lambda_smooth=0.0010,
    lambda_roll=0.0050,
    roll_integration="euler",
)
result4_configs["e4_ccg_phys_s0010_r0050_warm4"] = build_phys_config(
    ccg_base,
    "e4_ccg_phys_s0010_r0050_warm4",
    lambda_smooth=0.0010,
    lambda_roll=0.0050,
    warmup_epochs=4,
)
result4_configs["e4_ccg_phys_s0010_r0050_warm12"] = build_phys_config(
    ccg_base,
    "e4_ccg_phys_s0010_r0050_warm12",
    lambda_smooth=0.0010,
    lambda_roll=0.0050,
    warmup_epochs=12,
)

# ------------------------------------------------------------------
# VMD 系统扫描组
# ------------------------------------------------------------------
vmd_specs = {
    "e4_vmd_ccg_l005_lr1e3_mix": dict(lambda_vmd=0.05, lr=1.0e-3, use_state_mixer=True, warmup_epochs=5),
    "e4_vmd_ccg_l005_lr5e4_mix": dict(lambda_vmd=0.05, lr=5.0e-4, use_state_mixer=True, warmup_epochs=5),
    "e4_vmd_ccg_l005_lr1e3_nomix": dict(lambda_vmd=0.05, lr=1.0e-3, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l005_lr5e4_nomix": dict(lambda_vmd=0.05, lr=5.0e-4, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l005_lr3e4_nomix": dict(lambda_vmd=0.05, lr=3.0e-4, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l005_lr7e4_nomix": dict(lambda_vmd=0.05, lr=7.0e-4, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l002_lr5e4_nomix": dict(lambda_vmd=0.02, lr=5.0e-4, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l003_lr5e4_nomix": dict(lambda_vmd=0.03, lr=5.0e-4, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l008_lr5e4_nomix": dict(lambda_vmd=0.08, lr=5.0e-4, use_state_mixer=False, warmup_epochs=5),
    "e4_vmd_ccg_l005_lr5e4_nomix_warm3": dict(lambda_vmd=0.05, lr=5.0e-4, use_state_mixer=False, warmup_epochs=3),
    "e4_vmd_ccg_l005_lr5e4_nomix_warm8": dict(lambda_vmd=0.05, lr=5.0e-4, use_state_mixer=False, warmup_epochs=8),
    "e4_vmd_ccg_l005_lr5e4_h8_d01": dict(
        lambda_vmd=0.05,
        lr=5.0e-4,
        use_state_mixer=True,
        state_mixer_hidden_dim=8,
        state_mixer_dropout=0.1,
        warmup_epochs=5,
    ),
    "e4_vmd_ccg_l005_lr5e4_h4_d01": dict(
        lambda_vmd=0.05,
        lr=5.0e-4,
        use_state_mixer=True,
        state_mixer_hidden_dim=4,
        state_mixer_dropout=0.1,
        warmup_epochs=5,
    ),
    "e4_vmd_ccg_l005_lr5e4_nomix_a1500": dict(
        lambda_vmd=0.05,
        lr=5.0e-4,
        use_state_mixer=False,
        warmup_epochs=5,
        alpha=1500,
    ),
    "e4_vmd_ccg_l005_lr5e4_nomix_a2500": dict(
        lambda_vmd=0.05,
        lr=5.0e-4,
        use_state_mixer=False,
        warmup_epochs=5,
        alpha=2500,
    ),
}
for stem, kwargs in vmd_specs.items():
    result4_configs[stem] = build_vmd_config(vmd_base, stem, **kwargs)

# ------------------------------------------------------------------
# 联合验证组
# ------------------------------------------------------------------
joint_specs = {
    "e4_joint_l005_lr5e4_nomix_r0025": dict(lambda_smooth=0.0, lambda_roll=0.0025),
    "e4_joint_l005_lr5e4_nomix_r0050": dict(lambda_smooth=0.0, lambda_roll=0.0050),
    "e4_joint_l005_lr5e4_nomix_s0005_r0025": dict(lambda_smooth=0.0005, lambda_roll=0.0025),
    "e4_joint_l005_lr5e4_nomix_s0005_r0050": dict(lambda_smooth=0.0005, lambda_roll=0.0050),
    "e4_joint_l005_lr5e4_nomix_s0010_r0050": dict(lambda_smooth=0.0010, lambda_roll=0.0050),
    "e4_joint_l005_lr5e4_nomix_s0010_r0075": dict(lambda_smooth=0.0010, lambda_roll=0.0075),
}
for stem, kwargs in joint_specs.items():
    result4_configs[stem] = build_joint_config(vmd_base, stem, **kwargs)

# ------------------------------------------------------------------
# 多 seed 组：从当前默认候选克隆，并把 train.seed 改成指定值。
# ------------------------------------------------------------------
for base_stem in robust_bases:
    if base_stem not in result4_configs:
        raise ValueError(
            f"Unknown robust base config: {base_stem!r}. "
            f"Expected one of: {sorted(result4_configs)}"
        )
    base_cfg = result4_configs[base_stem]
    for seed in robust_seeds:
        robust_stem = f"{base_stem}_seed{seed}"
        robust_cfg = copy.deepcopy(base_cfg)
        robust_cfg["run_name"] = robust_stem
        robust_cfg.setdefault("train", {})["seed"] = int(seed)
        result4_configs[robust_stem] = robust_cfg

runtime_config_root.mkdir(parents=True, exist_ok=True)
for stem, cfg in result4_configs.items():
    save_yaml(runtime_config_root / f"{stem}.yaml", cfg)
    print(runtime_config_root / f"{stem}.yaml")
PY

trim_csv_into_array "$ROBUST_SEEDS" ROBUST_SEED_VALUES
trim_csv_into_array "$ROBUST_BASES" ROBUST_BASE_VALUES

BASELINE_CONFIGS=(
  "e4_ccg_xlstm_base"
)

PHYS_MAIN_CONFIGS=(
  "e4_ccg_phys_s0005_r0000"
  "e4_ccg_phys_s0010_r0000"
  "e4_ccg_phys_s0000_r0025"
  "e4_ccg_phys_s0000_r0050"
  "e4_ccg_phys_s0000_r0075"
  "e4_ccg_phys_s0005_r0025"
  "e4_ccg_phys_s0005_r0050"
  "e4_ccg_phys_s0010_r0050"
  "e4_ccg_phys_s0010_r0075"
  "e4_ccg_phys_s0010_r0100"
)

PHYS_DIAG_CONFIGS=(
  "e4_ccg_phys_s0010_r0050_normoff"
  "e4_ccg_phys_s0010_r0050_euler"
  "e4_ccg_phys_s0010_r0050_warm4"
  "e4_ccg_phys_s0010_r0050_warm12"
)

VMD_MAIN_CONFIGS=(
  "e4_vmd_ccg_l005_lr1e3_mix"
  "e4_vmd_ccg_l005_lr5e4_mix"
  "e4_vmd_ccg_l005_lr1e3_nomix"
  "e4_vmd_ccg_l005_lr5e4_nomix"
  "e4_vmd_ccg_l005_lr3e4_nomix"
  "e4_vmd_ccg_l005_lr7e4_nomix"
  "e4_vmd_ccg_l002_lr5e4_nomix"
  "e4_vmd_ccg_l003_lr5e4_nomix"
  "e4_vmd_ccg_l008_lr5e4_nomix"
  "e4_vmd_ccg_l005_lr5e4_nomix_warm3"
  "e4_vmd_ccg_l005_lr5e4_nomix_warm8"
)

VMD_DIAG_CONFIGS=(
  "e4_vmd_ccg_l005_lr5e4_h8_d01"
  "e4_vmd_ccg_l005_lr5e4_h4_d01"
  "e4_vmd_ccg_l005_lr5e4_nomix_a1500"
  "e4_vmd_ccg_l005_lr5e4_nomix_a2500"
)

JOINT_CONFIGS=(
  "e4_joint_l005_lr5e4_nomix_r0025"
  "e4_joint_l005_lr5e4_nomix_r0050"
  "e4_joint_l005_lr5e4_nomix_s0005_r0025"
  "e4_joint_l005_lr5e4_nomix_s0005_r0050"
  "e4_joint_l005_lr5e4_nomix_s0010_r0050"
  "e4_joint_l005_lr5e4_nomix_s0010_r0075"
)

ROBUST_CONFIGS=()
for base_stem in "${ROBUST_BASE_VALUES[@]}"; do
  for seed in "${ROBUST_SEED_VALUES[@]}"; do
    ROBUST_CONFIGS+=("${base_stem}_seed${seed}")
  done
done

PHYS_CONFIGS=("${BASELINE_CONFIGS[@]}" "${PHYS_MAIN_CONFIGS[@]}" "${PHYS_DIAG_CONFIGS[@]}")
VMD_CONFIGS=("${BASELINE_CONFIGS[@]}" "${VMD_MAIN_CONFIGS[@]}" "${VMD_DIAG_CONFIGS[@]}")
JOINT_GROUP_CONFIGS=("${BASELINE_CONFIGS[@]}" "${JOINT_CONFIGS[@]}")

ALL_CONFIGS=(
  "${BASELINE_CONFIGS[@]}"
  "${PHYS_MAIN_CONFIGS[@]}"
  "${PHYS_DIAG_CONFIGS[@]}"
  "${VMD_MAIN_CONFIGS[@]}"
  "${VMD_DIAG_CONFIGS[@]}"
  "${JOINT_CONFIGS[@]}"
)

SELECTED_CONFIGS=()
case "$RUN_GROUP" in
  full)
    SELECTED_CONFIGS=("${ALL_CONFIGS[@]}" "${ROBUST_CONFIGS[@]}")
    ;;
  all)
    SELECTED_CONFIGS=("${ALL_CONFIGS[@]}")
    ;;
  baseline)
    SELECTED_CONFIGS=("${BASELINE_CONFIGS[@]}")
    ;;
  phys_main)
    SELECTED_CONFIGS=("${BASELINE_CONFIGS[@]}" "${PHYS_MAIN_CONFIGS[@]}")
    ;;
  phys_diag)
    SELECTED_CONFIGS=("${BASELINE_CONFIGS[@]}" "${PHYS_DIAG_CONFIGS[@]}")
    ;;
  phys)
    SELECTED_CONFIGS=("${PHYS_CONFIGS[@]}")
    ;;
  vmd_main)
    SELECTED_CONFIGS=("${BASELINE_CONFIGS[@]}" "${VMD_MAIN_CONFIGS[@]}")
    ;;
  vmd_diag)
    SELECTED_CONFIGS=("${BASELINE_CONFIGS[@]}" "${VMD_DIAG_CONFIGS[@]}")
    ;;
  vmd)
    SELECTED_CONFIGS=("${VMD_CONFIGS[@]}")
    ;;
  joint)
    SELECTED_CONFIGS=("${JOINT_GROUP_CONFIGS[@]}")
    ;;
  robust)
    SELECTED_CONFIGS=("${ROBUST_CONFIGS[@]}")
    ;;
  *)
    echo "[Error] Unsupported RUN_GROUP=$RUN_GROUP. Expected full|all|baseline|phys_main|phys_diag|phys|vmd_main|vmd_diag|vmd|joint|robust." >&2
    exit 1
    ;;
esac

if [[ -n "$RUN_ONLY" ]]; then
  trim_csv_into_array "$RUN_ONLY" SELECTED_CONFIGS
fi

if [[ ${#SELECTED_CONFIGS[@]} -eq 0 ]]; then
  echo "[Error] No runtime configs were selected. Please check RUN_GROUP / RUN_ONLY." >&2
  exit 1
fi

for cfg in "${SELECTED_CONFIGS[@]}"; do
  if [[ ! -f "$RUNTIME_CONFIG_ROOT/${cfg}.yaml" ]]; then
    echo "[Error] Runtime config not found: $RUNTIME_CONFIG_ROOT/${cfg}.yaml" >&2
    exit 1
  fi
done

print_banner "[Selected Configs] ${SELECTED_CONFIGS[*]}"

declare -a CACHE_BUILDERS=()
add_unique_cache_builder() {
  local candidate="$1"
  if ! array_contains "$candidate" "${CACHE_BUILDERS[@]}"; then
    CACHE_BUILDERS+=("$candidate")
  fi
}

for cfg in "${SELECTED_CONFIGS[@]}"; do
  case "$cfg" in
    e4_vmd_*_a1500|e4_joint_*_a1500|*_a1500_seed*)
      add_unique_cache_builder "e4_vmd_ccg_l005_lr5e4_nomix_a1500"
      ;;
    e4_vmd_*_a2500|e4_joint_*_a2500|*_a2500_seed*)
      add_unique_cache_builder "e4_vmd_ccg_l005_lr5e4_nomix_a2500"
      ;;
    e4_vmd_*|e4_joint_*)
      add_unique_cache_builder "e4_vmd_ccg_l005_lr1e3_mix"
      ;;
  esac
done

for cache_cfg in "${CACHE_BUILDERS[@]}"; do
  run_cmd "build_cache_${cache_cfg}" "vmd_cache" \
    "$PYTHON_BIN" -m ship_motion.data.vmd --config "$RUNTIME_CONFIG_ROOT/${cache_cfg}.yaml"
done

for cfg in "${SELECTED_CONFIGS[@]}"; do
  run_cmd "train_${cfg}" "train" \
    "$PYTHON_BIN" -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/${cfg}.yaml"
done

SUMMARY_RUNS=()
for cfg in "${SELECTED_CONFIGS[@]}"; do
  if array_contains "$cfg" "${ALL_CONFIGS[@]}"; then
    SUMMARY_RUNS+=("$cfg")
  fi
done

if [[ "$SKIP_SUMMARY" != "1" ]]; then
  if [[ ${#SUMMARY_RUNS[@]} -gt 0 ]]; then
    RUNS_CSV=$(IFS=, ; echo "${SUMMARY_RUNS[*]}")
    run_cmd "summarize_${RESULT_NAME}" "summary" \
      "$PYTHON_BIN" -m ship_motion.summarize_results \
      --ablation-config "$ABLATION_CONFIG" \
      --runs "$RUNS_CSV" \
      --run-output-root "$OUTPUT_ROOT" \
      --summary-output-dir "$SUMMARY_ROOT"
  else
    echo "[Info] 当前选择的配置不在 configs/ablation_result4.yaml 中，跳过默认主表汇总。"
  fi
fi

PIPELINE_END_TS=$(date +%s)
print_banner "[Run Done] Result4 experiment pipeline finished | result_dir=$RESULT_DIR | total_elapsed=$(format_duration "$((PIPELINE_END_TS - PIPELINE_START_TS))")"
echo "[Next] 单 seed 主实验建议优先查看："
echo "  $SUMMARY_ROOT/ablation_routine_test.csv"
echo "  $SUMMARY_ROOT/ablation_ood_test.csv"
echo "  $SUMMARY_ROOT/physics_metrics.csv"
if [[ ${#ROBUST_CONFIGS[@]} -gt 0 ]]; then
  echo "[Note] robust 组默认只保留各 seed 的独立 run 输出，当前不会自动写入单 seed 主表。"
fi

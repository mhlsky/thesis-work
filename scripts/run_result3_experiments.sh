#!/usr/bin/env bash
set -euo pipefail

# 实验3：Phys / VMD 独立验证（Linux 云服务器版）。
#
# 设计目标：
# 1. 不修改核心训练代码，只通过“运行时配置改写”组织实验；
# 2. 把 Phys 与 VMD 分成两组独立验证，避免继续把两者耦合在一起解释；
# 3. 所有输出统一写到 results/result3/ 下，便于和 result_1 / result_2_followup 对照；
# 4. 通过把 runtime config 放在仓库根目录附近，绕开当前 default_run_dir 的双层嵌套问题。
#
# 默认会跑完整实验3。
# 可选环境变量：
# - RUN_GROUP=all|phys|vmd     只跑某一组；默认 all
# - RUN_ONLY=cfg1,cfg2         只跑指定 runtime config（不带 .yaml），优先级高于 RUN_GROUP
# - PYTHON_BIN=python          指定解释器，默认直接用当前激活环境中的 python
# - SKIP_SUMMARY=1             跳过最后的汇总
# - TRAIN_GPU_IDS=0,1          训练阶段允许调度的 GPU 列表；默认使用 0,1
# - TRAIN_MAX_CONCURRENT=2     训练阶段最大并发任务数；默认等于 GPU 数
#
# 用法：
#   conda activate your_env
#   bash scripts/run_result3_experiments.sh . result3
#   RUN_GROUP=phys bash scripts/run_result3_experiments.sh . result3

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result3}"
RUN_GROUP="${RUN_GROUP:-all}"
RUN_ONLY="${RUN_ONLY:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/server_train_env.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="$PROJECT_ROOT/results/$RESULT_NAME"
OUTPUT_ROOT="$RESULT_DIR/outputs"
LOG_ROOT="$RESULT_DIR/logs"
SUMMARY_ROOT="$OUTPUT_ROOT/summary"
RUNTIME_CONFIG_ROOT="$PROJECT_ROOT/runtime_configs_${RESULT_NAME}"
ABLATION_CONFIG="$PROJECT_ROOT/configs/ablation_result3.yaml"

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

reorder_configs_by_estimated_cost() {
  local runtime_root="$1"
  shift
  "$PYTHON_BIN" - "$runtime_root" "$@" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

runtime_root = Path(sys.argv[1])
config_names = sys.argv[2:]

def estimate_cost(cfg: dict) -> float:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    model_name = str(model_cfg.get("name", "")).strip().lower()
    epochs = float(train_cfg.get("epochs", 1) or 1)
    seq_len = float(data_cfg.get("seq_len", 128) or 128)
    pred_len = float(data_cfg.get("pred_len", 10) or 10)
    num_layers = float(model_cfg.get("num_layers", 1) or 1)

    score = epochs
    score *= max(seq_len / 128.0, 0.5)
    score *= max(pred_len / 10.0, 0.5)
    score *= 1.0 + 0.15 * max(num_layers - 1.0, 0.0)

    if model_name == "persistence":
        score *= 0.05
    elif model_name in {"lstm", "gru", "tcn"}:
        score *= 0.9
    elif model_name == "transformer":
        score *= 1.15
    elif model_name in {"lite_xlstm", "ccg_xlstm"}:
        score *= 1.2
    elif model_name == "vmd_ccg_xlstm":
        score *= 1.45

    vmd_cfg = cfg.get("vmd", {})
    if isinstance(vmd_cfg, dict) and bool(vmd_cfg.get("enabled", False)):
        score *= 1.15 + 0.05 * max(float(vmd_cfg.get("K", 3) or 3) - 3.0, 0.0)

    physics_cfg = cfg.get("physics", {})
    if isinstance(physics_cfg, dict) and bool(physics_cfg.get("enabled", False)):
        score *= 1.12

    if bool(model_cfg.get("use_state_mixer", False)):
        score *= 1.03

    return float(score)

ranked: list[tuple[float, str]] = []
for name in config_names:
    path = runtime_root / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ranked.append((estimate_cost(cfg), name))

for _, name in sorted(ranked, key=lambda item: (-item[0], item[1])):
    print(name)
PY
}

TRAIN_GPU_IDS_TEXT="${TRAIN_GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1}}"
TRAIN_MAX_CONCURRENT="${TRAIN_MAX_CONCURRENT:-0}"
trim_csv_into_array "$TRAIN_GPU_IDS_TEXT" TRAIN_GPU_IDS_ARRAY
if [[ ${#TRAIN_GPU_IDS_ARRAY[@]} -eq 0 ]]; then
  echo "[Error] TRAIN_GPU_IDS is empty after parsing." >&2
  exit 1
fi
if [[ "$TRAIN_MAX_CONCURRENT" == "0" ]]; then
  TRAIN_MAX_CONCURRENT=${#TRAIN_GPU_IDS_ARRAY[@]}
fi
if (( TRAIN_MAX_CONCURRENT <= 0 )); then
  echo "[Error] TRAIN_MAX_CONCURRENT must be a positive integer." >&2
  exit 1
fi

PER_JOB_CPU_CORES=$(( SHIP_MOTION_CPU_CORES / TRAIN_MAX_CONCURRENT ))
if (( PER_JOB_CPU_CORES < 1 )); then
  PER_JOB_CPU_CORES=1
fi
PER_JOB_SYSTEM_MEMORY_GB=$(( SHIP_MOTION_SYSTEM_MEMORY_GB / TRAIN_MAX_CONCURRENT ))
if (( PER_JOB_SYSTEM_MEMORY_GB < 1 )); then
  PER_JOB_SYSTEM_MEMORY_GB=1
fi

ACTIVE_PIDS=()
ACTIVE_GPUS=()
ACTIVE_CFGS=()

terminate_active_jobs() {
  local pid
  for pid in "${ACTIVE_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for pid in "${ACTIVE_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}

acquire_free_gpu() {
  while true; do
    local gpu_id
    for gpu_id in "${TRAIN_GPU_IDS_ARRAY[@]}"; do
      local in_use=0
      local active_gpu
      for active_gpu in "${ACTIVE_GPUS[@]}"; do
        if [[ "$active_gpu" == "$gpu_id" ]]; then
          in_use=1
          break
        fi
      done
      if (( in_use == 0 )); then
        printf '%s\n' "$gpu_id"
        return 0
      fi
    done
    wait_for_one_training_job
  done
}

wait_for_one_training_job() {
  while true; do
    local idx
    for idx in "${!ACTIVE_PIDS[@]}"; do
      local pid="${ACTIVE_PIDS[$idx]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        local gpu_id="${ACTIVE_GPUS[$idx]}"
        local cfg_name="${ACTIVE_CFGS[$idx]}"
        local status=0
        if wait "$pid"; then
          status=0
        else
          status=$?
        fi
        unset 'ACTIVE_PIDS[idx]' 'ACTIVE_GPUS[idx]' 'ACTIVE_CFGS[idx]'
        ACTIVE_PIDS=("${ACTIVE_PIDS[@]}")
        ACTIVE_GPUS=("${ACTIVE_GPUS[@]}")
        ACTIVE_CFGS=("${ACTIVE_CFGS[@]}")
        if (( status != 0 )); then
          echo "[Error] Training failed: cfg=$cfg_name | gpu=$gpu_id | exit_code=$status" >&2
          terminate_active_jobs
          exit "$status"
        fi
        return 0
      fi
    done
    sleep 5
  done
}

launch_training_job() {
  local cfg_name="$1"
  local gpu_id="$2"
  local log_name="train_${cfg_name}"
  local log_path="$LOG_ROOT/${log_name}.log"
  (
    set -o pipefail
    task_start_ts=$(date +%s)
    {
      print_banner "[Task Start] name=$log_name | stage=train | gpu=$gpu_id | started_at=$(date '+%F %T')"
      export CUDA_VISIBLE_DEVICES="$gpu_id"
      export SHIP_MOTION_CPU_CORES="$PER_JOB_CPU_CORES"
      export SHIP_MOTION_SYSTEM_MEMORY_GB="$PER_JOB_SYSTEM_MEMORY_GB"
      "$PYTHON_BIN" -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/${cfg_name}.yaml"
      task_end_ts=$(date +%s)
      print_banner "[Task Done] name=$log_name | stage=train | gpu=$gpu_id | elapsed=$(format_duration "$((task_end_ts - task_start_ts))") | log=$log_path"
    } 2>&1 | tee "$log_path"
  ) &
  ACTIVE_PIDS+=("$!")
  ACTIVE_GPUS+=("$gpu_id")
  ACTIVE_CFGS+=("$cfg_name")
}

print_banner "[Run Start] Result3 experiment pipeline | project_root=$PROJECT_ROOT | result_name=$RESULT_NAME | run_group=$RUN_GROUP"

run_cmd "prepare_result3_runtime_configs" "prepare_configs" \
  "$PYTHON_BIN" - "$PROJECT_ROOT" "$RESULT_NAME" "$RUNTIME_CONFIG_ROOT" <<'PY'
from __future__ import annotations

import copy
import sys
from pathlib import Path

import yaml

project_root = Path(sys.argv[1]).resolve()
result_name = sys.argv[2]
runtime_config_root = Path(sys.argv[3]).resolve()

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


ccg_base = load_yaml(configs_root / "ccg_xlstm.yaml")
vmd_base = load_yaml(configs_root / "vmd_ccg_xlstm.yaml")

result3_configs: dict[str, dict] = {}

# ------------------------------------------------------------------
# Phys 独立验证组：在当前最稳的 CCG-xLSTM 上单独加 physics。
# ------------------------------------------------------------------
cfg = with_common_output(ccg_base, "e3_ccg_xlstm_base")
result3_configs["e3_ccg_xlstm_base"] = cfg

phys_common = {
    "enabled": True,
    "dt": 1.0,
    "warmup_epochs": 8,
    "smoothness_target_cols": ["p", "r", "phi"],
    "normalize_by_target_std": True,
    "roll_integration": "trapezoid",
    "p_col": "p",
    "phi_col": "phi",
}

cfg = with_common_output(ccg_base, "e3_ccg_phys_s001_r000")
cfg["physics"] = dict(phys_common, lambda_smooth=0.001, lambda_roll=0.0)
result3_configs["e3_ccg_phys_s001_r000"] = cfg

cfg = with_common_output(ccg_base, "e3_ccg_phys_s000_r005")
cfg["physics"] = dict(phys_common, lambda_smooth=0.0, lambda_roll=0.005)
result3_configs["e3_ccg_phys_s000_r005"] = cfg

cfg = with_common_output(ccg_base, "e3_ccg_phys_s001_r005")
cfg["physics"] = dict(phys_common, lambda_smooth=0.001, lambda_roll=0.005)
result3_configs["e3_ccg_phys_s001_r005"] = cfg

cfg = with_common_output(ccg_base, "e3_ccg_phys_s001_r010")
cfg["physics"] = dict(phys_common, lambda_smooth=0.001, lambda_roll=0.01)
result3_configs["e3_ccg_phys_s001_r010"] = cfg

# ------------------------------------------------------------------
# VMD 独立验证组：先固定一组低权重 VMD 对照，再分别改四个方向。
# 四个方向：
# 1. warmup
# 2. learning rate
# 3. VMD 分解参数（K / alpha）
# 4. 去掉 state mixer，检查是否是 VMD 与后耦合结构交互不稳
# ------------------------------------------------------------------
vmd_low_weight_common = {
    "enabled": True,
    "K": 3,
    "alpha": 2000,
    "tau": 0.0,
    "DC": False,
    "init": 1,
    "tol": 1.0e-7,
    "lambda_vmd": 0.05,
    "warmup_epochs": 5,
    "cache_root": f"results/{result_name}/outputs/cache/vmd/K3_alpha2000",
}

cfg = with_common_output(vmd_base, "e3_vmd_ccg_l005_base")
cfg["vmd"] = copy.deepcopy(vmd_low_weight_common)
result3_configs["e3_vmd_ccg_l005_base"] = cfg

cfg = with_common_output(vmd_base, "e3_vmd_ccg_l005_warm10")
cfg["vmd"] = dict(vmd_low_weight_common, warmup_epochs=10)
result3_configs["e3_vmd_ccg_l005_warm10"] = cfg

cfg = with_common_output(vmd_base, "e3_vmd_ccg_l005_lr5e4")
cfg["vmd"] = copy.deepcopy(vmd_low_weight_common)
cfg.setdefault("train", {})["lr"] = 5.0e-4
result3_configs["e3_vmd_ccg_l005_lr5e4"] = cfg

cfg = with_common_output(vmd_base, "e3_vmd_ccg_l005_k2_a1000")
cfg["vmd"] = dict(
    vmd_low_weight_common,
    K=2,
    alpha=1000,
    cache_root=f"results/{result_name}/outputs/cache/vmd/K2_alpha1000",
)
result3_configs["e3_vmd_ccg_l005_k2_a1000"] = cfg

cfg = with_common_output(vmd_base, "e3_vmd_ccg_l005_k4_a3000")
cfg["vmd"] = dict(
    vmd_low_weight_common,
    K=4,
    alpha=3000,
    cache_root=f"results/{result_name}/outputs/cache/vmd/K4_alpha3000",
)
result3_configs["e3_vmd_ccg_l005_k4_a3000"] = cfg

cfg = with_common_output(vmd_base, "e3_vmd_ccg_l005_nomixer")
cfg["vmd"] = copy.deepcopy(vmd_low_weight_common)
cfg.setdefault("model", {})["use_state_mixer"] = False
result3_configs["e3_vmd_ccg_l005_nomixer"] = cfg

runtime_config_root.mkdir(parents=True, exist_ok=True)
for stem, cfg in result3_configs.items():
    save_yaml(runtime_config_root / f"{stem}.yaml", cfg)
    print(runtime_config_root / f"{stem}.yaml")
PY

PHYS_CONFIGS=(
  "e3_ccg_xlstm_base"
  "e3_ccg_phys_s001_r000"
  "e3_ccg_phys_s000_r005"
  "e3_ccg_phys_s001_r005"
  "e3_ccg_phys_s001_r010"
)

VMD_CONFIGS=(
  "e3_ccg_xlstm_base"
  "e3_vmd_ccg_l005_base"
  "e3_vmd_ccg_l005_warm10"
  "e3_vmd_ccg_l005_lr5e4"
  "e3_vmd_ccg_l005_k2_a1000"
  "e3_vmd_ccg_l005_k4_a3000"
  "e3_vmd_ccg_l005_nomixer"
)

ALL_CONFIGS=(
  "e3_ccg_xlstm_base"
  "e3_ccg_phys_s001_r000"
  "e3_ccg_phys_s000_r005"
  "e3_ccg_phys_s001_r005"
  "e3_ccg_phys_s001_r010"
  "e3_vmd_ccg_l005_base"
  "e3_vmd_ccg_l005_warm10"
  "e3_vmd_ccg_l005_lr5e4"
  "e3_vmd_ccg_l005_k2_a1000"
  "e3_vmd_ccg_l005_k4_a3000"
  "e3_vmd_ccg_l005_nomixer"
)

SELECTED_CONFIGS=()
case "$RUN_GROUP" in
  all)
    SELECTED_CONFIGS=("${ALL_CONFIGS[@]}")
    ;;
  phys)
    SELECTED_CONFIGS=("${PHYS_CONFIGS[@]}")
    ;;
  vmd)
    SELECTED_CONFIGS=("${VMD_CONFIGS[@]}")
    ;;
  *)
    echo "[Error] Unsupported RUN_GROUP=$RUN_GROUP. Expected all|phys|vmd." >&2
    exit 1
    ;;
esac

if [[ -n "$RUN_ONLY" ]]; then
  IFS=',' read -r -a SELECTED_CONFIGS <<< "$RUN_ONLY"
fi

print_banner "[Selected Configs] ${SELECTED_CONFIGS[*]}"
print_banner "[Train Parallelism] gpu_ids=${TRAIN_GPU_IDS_ARRAY[*]} | max_concurrent=$TRAIN_MAX_CONCURRENT | per_job_cpu_cores=$PER_JOB_CPU_CORES | per_job_memory_gb=$PER_JOB_SYSTEM_MEMORY_GB"

# 需要 VMD cache 的配置只按唯一 K/alpha 组合构建一次。
declare -a CACHE_CONFIGS=()
for cfg in "${SELECTED_CONFIGS[@]}"; do
  case "$cfg" in
    e3_vmd_ccg_l005_base|e3_vmd_ccg_l005_warm10|e3_vmd_ccg_l005_lr5e4|e3_vmd_ccg_l005_nomixer)
      if [[ ! " ${CACHE_CONFIGS[*]} " =~ " e3_vmd_ccg_l005_base " ]]; then
        CACHE_CONFIGS+=("e3_vmd_ccg_l005_base")
      fi
      ;;
    e3_vmd_ccg_l005_k2_a1000)
      CACHE_CONFIGS+=("e3_vmd_ccg_l005_k2_a1000")
      ;;
    e3_vmd_ccg_l005_k4_a3000)
      CACHE_CONFIGS+=("e3_vmd_ccg_l005_k4_a3000")
      ;;
  esac
done

for cache_cfg in "${CACHE_CONFIGS[@]}"; do
  run_cmd "build_cache_${cache_cfg}" "vmd_cache" \
    "$PYTHON_BIN" -m ship_motion.data.vmd --config "$RUNTIME_CONFIG_ROOT/${cache_cfg}.yaml"
done

mapfile -t SELECTED_CONFIGS < <(reorder_configs_by_estimated_cost "$RUNTIME_CONFIG_ROOT" "${SELECTED_CONFIGS[@]}")
print_banner "[Dispatch Order] ${SELECTED_CONFIGS[*]}"

for cfg in "${SELECTED_CONFIGS[@]}"; do
  while (( ${#ACTIVE_PIDS[@]} >= TRAIN_MAX_CONCURRENT )); do
    wait_for_one_training_job
  done
  assigned_gpu="$(acquire_free_gpu)"
  echo "[Dispatch] cfg=$cfg -> gpu=$assigned_gpu"
  launch_training_job "$cfg" "$assigned_gpu"
done

while (( ${#ACTIVE_PIDS[@]} > 0 )); do
  wait_for_one_training_job
done

if [[ "$SKIP_SUMMARY" != "1" ]]; then
  RUNS_CSV=$(IFS=, ; echo "${SELECTED_CONFIGS[*]}")
  run_cmd "summarize_${RESULT_NAME}" "summary" \
    "$PYTHON_BIN" -m ship_motion.summarize_results \
    --ablation-config "$ABLATION_CONFIG" \
    --runs "$RUNS_CSV" \
    --run-output-root "$OUTPUT_ROOT" \
    --summary-output-dir "$SUMMARY_ROOT"
fi

PIPELINE_END_TS=$(date +%s)
print_banner "[Run Done] Result3 experiment pipeline finished | result_dir=$RESULT_DIR | total_elapsed=$(format_duration "$((PIPELINE_END_TS - PIPELINE_START_TS))")"
echo "[Next] 建议优先查看："
echo "  $SUMMARY_ROOT/ablation_routine_test.csv"
echo "  $SUMMARY_ROOT/ablation_ood_test.csv"
echo "  $SUMMARY_ROOT/physics_metrics.csv"

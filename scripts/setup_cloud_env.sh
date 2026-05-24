#!/usr/bin/env bash
set -euo pipefail

# 云上环境初始化脚本。
# 用途：
# 1. 检查当前已激活的 Python 环境；
# 2. 复用该环境中的 torch / numpy / sklearn / vmdpy 等依赖；
# 3. 检查 GPU 与 torch CUDA；
# 4. 检查数据目录；
# 5. 运行最小 smoke，确认环境基本可用。
#
# 使用前提：
# - 你已经在云服务器上手动激活好了自己的 conda / venv 环境；
# - 如果该环境里缺少项目依赖，脚本会尝试直接安装到当前环境中。
# - 如果你希望缺少 torch 时安装指定版本/来源，可在运行前设置：
#   TORCH_INSTALL_SPEC='torch==你的版本'

PROJECT_ROOT="${1:-$PWD}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export SETUPTOOLS_USE_DISTUTILS="${SETUPTOOLS_USE_DISTUTILS:-stdlib}"
SETUP_START_TS=$(date +%s)
TORCH_INSTALL_SPEC="${TORCH_INSTALL_SPEC:-torch}"

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

print_stage() {
  local stage_name="$1"
  echo
  echo "--------------------------------------------------------------------------------"
  echo "[Stage] $stage_name"
  echo "--------------------------------------------------------------------------------"
}

run_check() {
  local name="$1"
  shift
  local task_start_ts
  task_start_ts=$(date +%s)
  print_banner "[Task Start] name=$name | started_at=$(date '+%F %T')"
  "$@"
  local task_end_ts
  task_end_ts=$(date +%s)
  print_banner "[Task Done] name=$name | elapsed=$(format_duration "$((task_end_ts - task_start_ts))")"
}

ensure_pip_available() {
  if python -m pip --version >/dev/null 2>&1; then
    return 0
  fi
  print_banner "[Info] pip not found in current environment, trying ensurepip"
  python -m ensurepip --upgrade
}

install_python_package() {
  local package_spec="$1"
  print_banner "[Install] package=$package_spec | target_python=$(python -c 'import sys; print(sys.executable)')"
  python -m pip install "$package_spec"
}

print_banner "[Run Start] Linux cloud environment setup | project_root=$PROJECT_ROOT"

print_stage "System information"
run_check "system_info" uname -a
run_check "python_path" python - <<'PY'
import sys
print(sys.executable)
PY
run_check "python_version" python --version

print_stage "Check Python environment"
ensure_pip_available
run_check "python_env_prepare" python - <<'PY'
import importlib
import sys

print("python:", sys.executable)
for module_name in ["pip", "setuptools"]:
    try:
        importlib.import_module(module_name)
        print(f"import ok: {module_name}")
    except ModuleNotFoundError:
        print(f"missing bootstrap module: {module_name}")
PY

declare -a module_package_pairs=(
  "torch:${TORCH_INSTALL_SPEC}"
  "numpy:numpy==1.26.4"
  "yaml:PyYAML>=6.0"
  "sklearn:scikit-learn>=1.5"
  "tqdm:tqdm>=4.66"
  "vmdpy:vmdpy>=0.2"
  "matplotlib:matplotlib>=3.9"
)

declare -a missing_package_specs=()
for pair in "${module_package_pairs[@]}"; do
  module_name="${pair%%:*}"
  package_spec="${pair#*:}"
  if python -c "import importlib; importlib.import_module('${module_name}')" >/dev/null 2>&1; then
    echo "[Dependency] OK module=${module_name}"
  else
    echo "[Dependency] MISSING module=${module_name} -> will install package=${package_spec}"
    missing_package_specs+=("${package_spec}")
  fi
done

if [ "${#missing_package_specs[@]}" -gt 0 ]; then
  print_stage "Install missing Python dependencies"
  for package_spec in "${missing_package_specs[@]}"; do
    run_check "install_$(echo "$package_spec" | tr '=><.-' '_' | tr -cd '[:alnum:]_')" install_python_package "$package_spec"
  done
else
  echo "==> All required Python dependencies are already present in the current environment."
fi

run_check "python_env_summary" python - <<'PY'
import importlib
import sys

module_names = ["torch", "numpy", "yaml", "sklearn", "tqdm", "vmdpy", "matplotlib"]
print("python:", sys.executable)
for module_name in module_names:
    module = importlib.import_module(module_name)
    print(f"import ok: {module_name} | version={getattr(module, '__version__', 'unknown')}")
PY

print_stage "Check GPU and Torch CUDA"
if command -v nvidia-smi >/dev/null 2>&1; then
  run_check "nvidia_smi" nvidia-smi
else
  echo "WARNING: nvidia-smi not found. If you expected GPU, please check the cloud runtime."
fi

run_check "torch_cuda_check" python - <<'PY'
import torch
import sys
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch_cuda:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY

print_stage "Check data and directories"
run_check "check_train_dir" test -d data/patrol_ship_routine/processed/train
run_check "check_val_dir" test -d data/patrol_ship_routine/processed/validation
run_check "check_routine_test_dir" test -d data/patrol_ship_routine/processed/test
run_check "check_ood_test_dir" test -d data/patrol_ship_ood/processed/test

run_check "prepare_output_dirs" mkdir -p outputs logs

print_stage "Run smoke checks"
run_check "smoke_dataset" python -m ship_motion.data.dataset --config configs/base.yaml --smoke
run_check "smoke_train_lstm" python -m ship_motion.train --config configs/lstm.yaml --smoke
run_check "smoke_vmd" python -m ship_motion.data.vmd --config configs/vmd_ccg_xlstm.yaml --smoke

SETUP_END_TS=$(date +%s)
print_banner "[Run Done] Linux cloud environment setup finished | total_elapsed=$(format_duration "$((SETUP_END_TS - SETUP_START_TS))")"

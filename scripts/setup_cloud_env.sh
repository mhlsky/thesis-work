#!/usr/bin/env bash
set -euo pipefail

# 云上环境初始化脚本。
# 用途：
# 1. 安装 / 检查 uv；
# 2. 同步项目依赖；
# 3. 检查 GPU 与 torch CUDA；
# 4. 检查数据目录；
# 5. 运行最小 smoke，确认环境基本可用。

PROJECT_ROOT="${1:-$PWD}"
cd "$PROJECT_ROOT"
SETUP_START_TS=$(date +%s)

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

print_banner "[Run Start] Linux cloud environment setup | project_root=$PROJECT_ROOT"

print_stage "System information"
run_check "system_info" uname -a
run_check "python_version" bash -lc 'python3 --version || true'

print_stage "Install and verify uv"
echo "==> Installing uv if needed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv version"
run_check "uv_version" uv --version

echo "==> Syncing dependencies"
run_check "uv_sync" uv sync

print_stage "Check GPU and Torch CUDA"
bash -lc 'nvidia-smi || echo "WARNING: nvidia-smi not found. If you expected GPU, please check the cloud runtime."'

run_check "torch_cuda_check" uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
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
run_check "smoke_dataset" uv run python -m ship_motion.data.dataset --config configs/base.yaml --smoke
run_check "smoke_train_lstm" uv run python -m ship_motion.train --config configs/lstm.yaml --smoke
run_check "smoke_vmd" uv run python -m ship_motion.data.vmd --config configs/vmd_ccg_xlstm.yaml --smoke

SETUP_END_TS=$(date +%s)
print_banner "[Run Done] Linux cloud environment setup finished | total_elapsed=$(format_duration "$((SETUP_END_TS - SETUP_START_TS))")"

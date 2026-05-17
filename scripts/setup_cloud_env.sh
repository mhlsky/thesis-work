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

echo "==> System info"
uname -a

echo "==> Python version"
python3 --version || true

echo "==> Installing uv if needed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> uv version"
uv --version

echo "==> Syncing dependencies"
uv sync

echo "==> Checking GPU"
nvidia-smi || echo "WARNING: nvidia-smi not found. If you expected GPU, please check the cloud runtime."

echo "==> Checking torch CUDA"
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY

echo "==> Checking required data directories"
test -d data/patrol_ship_routine/processed/train
test -d data/patrol_ship_routine/processed/validation
test -d data/patrol_ship_routine/processed/test
test -d data/patrol_ship_ood/processed/test

echo "==> Ensuring output/log directories"
mkdir -p outputs
mkdir -p logs

echo "==> Running smoke checks"
uv run python -m ship_motion.data.dataset --config configs/base.yaml --smoke
uv run python -m ship_motion.train --config configs/lstm.yaml --smoke
uv run python -m ship_motion.data.vmd --config configs/vmd_ccg_xlstm.yaml --smoke

echo "==> Cloud environment setup finished"

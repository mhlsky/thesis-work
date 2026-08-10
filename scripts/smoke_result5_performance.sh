#!/usr/bin/env bash
set -euo pipefail

# 实验5 smoke：只使用两个已有 checkpoint、CPU 与极少计时次数，验证性能测试全流程。
# 输出写入 outputs/smoke/result5_performance/，不会影响正式 results/ 目录。

PROJECT_ROOT="${1:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

RESULT_DIR="$PROJECT_ROOT/outputs/smoke/result5_performance"
LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR"

print_banner() {
  local message="$1"
  echo "================================================================================"
  echo "$message"
  echo "================================================================================"
}

# 本地 smoke 强制 CPU，避免把 CUDA 可用性当成工作流正确性的前置条件。
print_banner "[Smoke Start] Experiment5 performance smoke"
CUDA_VISIBLE_DEVICES="" "$PYTHON_BIN" -u -m ship_motion.benchmark_inference \
  --config "$PROJECT_ROOT/configs/experiment5_performance.yaml" \
  --output-dir "$RESULT_DIR" \
  --smoke 2>&1 | tee "$LOG_DIR/benchmark_inference.log"
print_banner "[Smoke Done] outputs at $RESULT_DIR"

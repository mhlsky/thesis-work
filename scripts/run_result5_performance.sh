#!/usr/bin/env bash
set -euo pipefail

# 实验5：已有模型的复杂度与单样本推理性能正式测试（Linux 云服务器版）。
#
# 本脚本不训练模型、不读取数据集评估，也不会改写已有 OOD 结果。
# 它只加载指定 checkpoint，测量 batch=1 的纯模型前向性能。
#
# 用法：
#   bash scripts/run_result5_performance.sh . result5_performance
#   BENCHMARK_PROFILES=cpu_fp32 bash scripts/run_result5_performance.sh . result5_cpu
#   BENCHMARK_PROFILES=cuda_fp32 bash scripts/run_result5_performance.sh . result5_cuda
#   BENCHMARK_MODELS=ccg_baseline,joint_final bash scripts/run_result5_performance.sh . result5_shortlist

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result5_performance}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BENCHMARK_PROFILES="${BENCHMARK_PROFILES:-cpu_fp32,cuda_fp32}"
BENCHMARK_MODELS="${BENCHMARK_MODELS:-}"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/server_train_env.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="$PROJECT_ROOT/results/$RESULT_NAME"
OUTPUT_DIR="$RESULT_DIR/outputs/performance"
LOG_DIR="$RESULT_DIR/logs"
CONFIG_PATH="$PROJECT_ROOT/configs/experiment5_performance.yaml"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

print_banner() {
  local message="$1"
  echo "================================================================================"
  echo "$message"
  echo "================================================================================"
}

run_cmd() {
  local name="$1"
  shift
  local started_at
  started_at=$(date +%s)
  print_banner "[Task Start] $name | started_at=$(date '+%F %T')"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  local ended_at
  ended_at=$(date +%s)
  print_banner "[Task Done] $name | elapsed=$((ended_at - started_at))s | log=$LOG_DIR/${name}.log"
}

print_banner "[Run Start] Experiment5 performance benchmark | result_name=$RESULT_NAME"
echo "[Run] profiles=$BENCHMARK_PROFILES"
if [[ -n "$BENCHMARK_MODELS" ]]; then
  echo "[Run] models=$BENCHMARK_MODELS"
else
  echo "[Run] models=<all>"
fi
echo "[Run] output_dir=$OUTPUT_DIR"

args=(
  -m ship_motion.benchmark_inference
  --config "$CONFIG_PATH"
  --output-dir "$OUTPUT_DIR"
  --profiles "$BENCHMARK_PROFILES"
)
if [[ -n "$BENCHMARK_MODELS" ]]; then
  args+=(--models "$BENCHMARK_MODELS")
fi

run_cmd "benchmark_inference" "$PYTHON_BIN" -u "${args[@]}"

print_banner "[Run Done] Result5 performance benchmark finished: $OUTPUT_DIR/model_benchmark.csv"

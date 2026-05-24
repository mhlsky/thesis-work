#!/usr/bin/env bash
set -euo pipefail

# 对已经跑完的正式结果目录重新执行汇总与绘图。
# 典型用途：
# 1. 训练都完成了，但想补生成 summary CSV；
# 2. 改了 plot_results.py 之后，重画正式图表；
# 3. persistence 补跑完成后，重新汇总完整主表。

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result_2}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="$PROJECT_ROOT/results/$RESULT_NAME"
OUTPUT_ROOT="$RESULT_DIR/outputs"
LOG_ROOT="$RESULT_DIR/logs"
ABLATION_CONFIG="$PROJECT_ROOT/configs/ablation_list.yaml"
FIG_NUM_WINDOWS="${FIG_NUM_WINDOWS:-200}"
FIG_HORIZON="${FIG_HORIZON:-0}"

mkdir -p "$LOG_ROOT"

if [[ ! -d "$OUTPUT_ROOT" ]]; then
  echo "[Error] Output root not found: $OUTPUT_ROOT" >&2
  exit 1
fi

run_cmd() {
  local name="$1"
  shift
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
}

run_cmd "summarize_results_regen" \
  python -m ship_motion.summarize_results \
  --ablation-config "$ABLATION_CONFIG" \
  --run-output-root "$OUTPUT_ROOT" \
  --summary-output-dir "$OUTPUT_ROOT/summary"

run_cmd "plot_results_regen" \
  python -m ship_motion.plot_results \
  --ablation-config "$ABLATION_CONFIG" \
  --run-output-root "$OUTPUT_ROOT" \
  --summary-output-dir "$OUTPUT_ROOT/summary" \
  --num-windows "$FIG_NUM_WINDOWS" \
  --horizon "$FIG_HORIZON"

echo "[Done] Regenerated summary and figures under: $OUTPUT_ROOT/summary"

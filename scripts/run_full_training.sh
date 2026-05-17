#!/usr/bin/env bash
set -euo pipefail

# 云上一键正式训练脚本。
# 用途：
# 1. 同步依赖；
# 2. 构建正式 VMD 缓存；
# 3. 顺序训练主线模型；
# 4. 导出 predictions；
# 5. 汇总结果并生成图表。

PROJECT_ROOT="${1:-$PWD}"
cd "$PROJECT_ROOT"

mkdir -p logs
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

print_stage() {
  local stage_name="$1"
  echo
  echo "--------------------------------------------------------------------------------"
  echo "[Stage] $stage_name"
  echo "--------------------------------------------------------------------------------"
}

run_cmd() {
  local name="$1"
  local stage_label="$2"
  shift
  shift
  local task_start_ts
  task_start_ts=$(date +%s)
  print_banner "[Task Start] name=$name | stage=$stage_label | started_at=$(date '+%F %T')"
  "$@" 2>&1 | tee "logs/${name}.log"
  local task_end_ts
  task_end_ts=$(date +%s)
  print_banner "[Task Done] name=$name | stage=$stage_label | elapsed=$(format_duration "$((task_end_ts - task_start_ts))") | log=logs/${name}.log"
}

print_banner "[Run Start] Full Linux training pipeline | project_root=$PROJECT_ROOT"

print_stage "Environment preparation"
echo "==> Ensuring uv is available"
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Syncing dependencies"
uv sync

print_stage "Step 03: build formal VMD cache"
run_cmd "step03_build_vmd_cache" "step03_vmd_cache" \
  uv run python -m ship_motion.data.vmd --config configs/vmd_ccg_phys_xlstm.yaml

print_stage "Step 02/04/05/06: train main models"
run_cmd "train_lstm" "train_baselines" \
  uv run python -m ship_motion.train --config configs/lstm.yaml

run_cmd "train_gru" "train_baselines" \
  uv run python -m ship_motion.train --config configs/gru.yaml

run_cmd "train_transformer" "train_baselines" \
  uv run python -m ship_motion.train --config configs/transformer.yaml

run_cmd "train_lite_xlstm" "train_xlstm" \
  uv run python -m ship_motion.train --config configs/lite_xlstm.yaml

run_cmd "train_ccg_xlstm" "train_xlstm" \
  uv run python -m ship_motion.train --config configs/ccg_xlstm.yaml

run_cmd "train_vmd_ccg_xlstm" "train_vmd_xlstm" \
  uv run python -m ship_motion.train --config configs/vmd_ccg_xlstm.yaml

run_cmd "train_vmd_ccg_phys_xlstm" "train_final_model" \
  uv run python -m ship_motion.train --config configs/vmd_ccg_phys_xlstm.yaml

print_stage "Step 07: export predictions"
run_cmd "eval_lstm" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/lstm.yaml --checkpoint outputs/lstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_gru" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/gru.yaml --checkpoint outputs/gru_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_transformer" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/transformer.yaml --checkpoint outputs/transformer_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_lite_xlstm" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/lite_xlstm.yaml --checkpoint outputs/lite_xlstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_ccg_xlstm" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/ccg_xlstm.yaml --checkpoint outputs/ccg_xlstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_vmd_ccg_xlstm" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/vmd_ccg_xlstm.yaml --checkpoint outputs/vmd_ccg_xlstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_vmd_ccg_phys_xlstm" "eval_predictions" \
  uv run python -m ship_motion.evaluate --config configs/vmd_ccg_phys_xlstm.yaml --checkpoint outputs/vmd_ccg_phys_xlstm_seq128_pred10/best.pt --split all --save-predictions

print_stage "Step 07: summarize and plot"
run_cmd "summarize_results" "step07_summary" \
  uv run python -m ship_motion.summarize_results --ablation-config configs/ablation_list.yaml --run-output-root outputs --summary-output-dir outputs/summary

run_cmd "plot_results" "step07_plot" \
  uv run python -m ship_motion.plot_results --ablation-config configs/ablation_list.yaml --run-output-root outputs --summary-output-dir outputs/summary

PIPELINE_END_TS=$(date +%s)
print_banner "[Run Done] Full Linux training pipeline finished | total_elapsed=$(format_duration "$((PIPELINE_END_TS - PIPELINE_START_TS))")"

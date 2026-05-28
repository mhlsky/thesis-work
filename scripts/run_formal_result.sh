#!/usr/bin/env bash
set -euo pipefail

# Linux 云上正式实验脚本。
# 目标：
# 1. 把本次正式实验的 outputs / logs 全部收拢到 results/<result_name>/ 下；
# 2. 自动为每个配置生成 runtime config，避免手工改 YAML；
# 3. 补跑 persistence，并顺序完成主线训练、评估、汇总与绘图；
# 4. 最终产出论文可用的 summary CSV 与 figures。

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result_2}"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/server_train_env.sh"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="$PROJECT_ROOT/results/$RESULT_NAME"
OUTPUT_ROOT="$RESULT_DIR/outputs"
LOG_ROOT="$RESULT_DIR/logs"
RUNTIME_CONFIG_ROOT="$RESULT_DIR/runtime_configs"
ABLATION_CONFIG="$PROJECT_ROOT/configs/ablation_list.yaml"

# 正式绘图默认只取前 200 个窗口、预测第 1 个 horizon。
FIG_NUM_WINDOWS="${FIG_NUM_WINDOWS:-200}"
FIG_HORIZON="${FIG_HORIZON:-0}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$RUNTIME_CONFIG_ROOT"
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
  shift 2
  local task_start_ts
  task_start_ts=$(date +%s)
  print_banner "[Task Start] name=$name | stage=$stage_label | started_at=$(date '+%F %T')"
  "$@" 2>&1 | tee "$LOG_ROOT/${name}.log"
  local task_end_ts
  task_end_ts=$(date +%s)
  print_banner "[Task Done] name=$name | stage=$stage_label | elapsed=$(format_duration "$((task_end_ts - task_start_ts))") | log=$LOG_ROOT/${name}.log"
}

print_banner "[Run Start] Formal Linux experiment pipeline | project_root=$PROJECT_ROOT | result_name=$RESULT_NAME"

print_stage "Prepare runtime configs"
run_cmd "prepare_runtime_configs" "prepare_configs" \
  python -m ship_motion.prepare_formal_result --result-name "$RESULT_NAME" --repo-root "$PROJECT_ROOT"

print_stage "Environment check"
run_cmd "env_check" "environment" python - <<'PY'
import sys
import torch

print(f"[Env] python: {sys.executable}")
print(f"[Env] torch: {torch.__version__}")
print(f"[Env] torch_cuda: {torch.version.cuda}")
print(f"[Env] cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Env] device_name: {torch.cuda.get_device_name(0)}")
PY

print_stage "Step 03: build formal VMD cache"
run_cmd "step03_build_vmd_cache" "step03_vmd_cache" \
  python -m ship_motion.data.vmd --config "$RUNTIME_CONFIG_ROOT/vmd_ccg_phys_xlstm.yaml"

print_stage "Step 02/04/05/06: train formal runs"
run_cmd "train_persistence" "train_baselines" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/persistence.yaml"

run_cmd "train_lstm" "train_baselines" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/lstm.yaml"

run_cmd "train_gru" "train_baselines" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/gru.yaml"

run_cmd "train_transformer" "train_baselines" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/transformer.yaml"

run_cmd "train_lite_xlstm" "train_xlstm" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/lite_xlstm.yaml"

run_cmd "train_ccg_xlstm" "train_xlstm" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/ccg_xlstm.yaml"

run_cmd "train_vmd_ccg_xlstm" "train_vmd_xlstm" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/vmd_ccg_xlstm.yaml"

run_cmd "train_vmd_ccg_phys_xlstm" "train_final_model" \
  python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/vmd_ccg_phys_xlstm.yaml"

print_stage "Step 07: export predictions for all formal runs"
run_cmd "eval_persistence" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/persistence.yaml" --checkpoint "$OUTPUT_ROOT/persistence_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_lstm" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/lstm.yaml" --checkpoint "$OUTPUT_ROOT/lstm_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_gru" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/gru.yaml" --checkpoint "$OUTPUT_ROOT/gru_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_transformer" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/transformer.yaml" --checkpoint "$OUTPUT_ROOT/transformer_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_lite_xlstm" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/lite_xlstm.yaml" --checkpoint "$OUTPUT_ROOT/lite_xlstm_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_ccg_xlstm" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/ccg_xlstm.yaml" --checkpoint "$OUTPUT_ROOT/ccg_xlstm_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_vmd_ccg_xlstm" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/vmd_ccg_xlstm.yaml" --checkpoint "$OUTPUT_ROOT/vmd_ccg_xlstm_seq128_pred10/best.pt" --split all --save-predictions

run_cmd "eval_vmd_ccg_phys_xlstm" "eval_predictions" \
  python -m ship_motion.evaluate --config "$RUNTIME_CONFIG_ROOT/vmd_ccg_phys_xlstm.yaml" --checkpoint "$OUTPUT_ROOT/vmd_ccg_phys_xlstm_seq128_pred10/best.pt" --split all --save-predictions

print_stage "Step 07: summarize and generate formal figures"
run_cmd "summarize_results" "step07_summary" \
  python -m ship_motion.summarize_results \
  --ablation-config "$ABLATION_CONFIG" \
  --run-output-root "$OUTPUT_ROOT" \
  --summary-output-dir "$OUTPUT_ROOT/summary"

run_cmd "plot_results" "step07_plot" \
  python -m ship_motion.plot_results \
  --ablation-config "$ABLATION_CONFIG" \
  --run-output-root "$OUTPUT_ROOT" \
  --summary-output-dir "$OUTPUT_ROOT/summary" \
  --num-windows "$FIG_NUM_WINDOWS" \
  --horizon "$FIG_HORIZON"

PIPELINE_END_TS=$(date +%s)
print_banner "[Run Done] Formal Linux experiment pipeline finished | result_dir=$RESULT_DIR | total_elapsed=$(format_duration "$((PIPELINE_END_TS - PIPELINE_START_TS))")"

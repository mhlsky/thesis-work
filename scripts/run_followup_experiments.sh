#!/usr/bin/env bash
set -euo pipefail

# 第一次正式实验后的 Linux 补充实验脚本。
# 目标：
# 1. 只重跑“result_1 之后真正受训练/评估逻辑改动影响”的模型；
# 2. 不重复训练未改动主干的 baseline / main 模型；
# 3. 把新增结果统一写到 results/<result_name>/ 下，便于和 result_1 做正式对照。
#
# 默认补充：
# - persistence：result_1 中缺正式结果，需要补齐主表；
# - vmd_ccg_xlstm：新增 VMD warmup，会改变训练行为；
# - vmd_ccg_phys_xlstm：新增 VMD/physics warmup 与更稳的 physics 配置解析。
#
# 可选环境变量：
# - SKIP_PERSISTENCE=1     跳过 persistence
# - INCLUDE_CCG_REPEAT=1   额外复现一次 ccg_xlstm，验证当前最好 OOD 结果的稳定性

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result_2_followup}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

RESULT_DIR="$PROJECT_ROOT/results/$RESULT_NAME"
OUTPUT_ROOT="$RESULT_DIR/outputs"
LOG_ROOT="$RESULT_DIR/logs"
RUNTIME_CONFIG_ROOT="$RESULT_DIR/runtime_configs"
SKIP_PERSISTENCE="${SKIP_PERSISTENCE:-0}"
INCLUDE_CCG_REPEAT="${INCLUDE_CCG_REPEAT:-0}"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$RUNTIME_CONFIG_ROOT"
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

declare -a SELECTED_CONFIGS=()
declare -a TRAIN_RUNTIME_CONFIGS=()
declare -a TRAIN_LOG_NAMES=()
declare -a TRAIN_STAGE_NAMES=()

if [[ "$SKIP_PERSISTENCE" != "1" ]]; then
  SELECTED_CONFIGS+=("configs/persistence.yaml")
  TRAIN_RUNTIME_CONFIGS+=("persistence.yaml")
  TRAIN_LOG_NAMES+=("train_persistence")
  TRAIN_STAGE_NAMES+=("followup_baseline")
fi

SELECTED_CONFIGS+=("configs/vmd_ccg_xlstm.yaml")
TRAIN_RUNTIME_CONFIGS+=("vmd_ccg_xlstm.yaml")
TRAIN_LOG_NAMES+=("train_vmd_ccg_xlstm")
TRAIN_STAGE_NAMES+=("followup_vmd")

SELECTED_CONFIGS+=("configs/vmd_ccg_phys_xlstm.yaml")
TRAIN_RUNTIME_CONFIGS+=("vmd_ccg_phys_xlstm.yaml")
TRAIN_LOG_NAMES+=("train_vmd_ccg_phys_xlstm")
TRAIN_STAGE_NAMES+=("followup_physics")

if [[ "$INCLUDE_CCG_REPEAT" == "1" ]]; then
  SELECTED_CONFIGS+=("configs/ccg_xlstm.yaml")
  TRAIN_RUNTIME_CONFIGS+=("ccg_xlstm.yaml")
  TRAIN_LOG_NAMES+=("train_ccg_xlstm_repeat")
  TRAIN_STAGE_NAMES+=("followup_repro")
fi

print_banner "[Run Start] Follow-up Linux experiments after result_1 | project_root=$PROJECT_ROOT | result_name=$RESULT_NAME"
echo "[Info] 默认不会重复训练 LSTM / GRU / Transformer / Lite-xLSTM / CCG-xLSTM（除非显式要求复现 CCG）。"

run_cmd "prepare_runtime_configs" "prepare_configs" \
  python -m ship_motion.prepare_formal_result --result-name "$RESULT_NAME" --repo-root "$PROJECT_ROOT" --configs "${SELECTED_CONFIGS[@]}"

# 只要本轮有 VMD 模型，就共享构建一次正式缓存即可。
run_cmd "step03_build_vmd_cache" "followup_vmd_cache" \
  python -m ship_motion.data.vmd --config "$RUNTIME_CONFIG_ROOT/vmd_ccg_phys_xlstm.yaml"

for idx in "${!TRAIN_RUNTIME_CONFIGS[@]}"; do
  runtime_config="${TRAIN_RUNTIME_CONFIGS[$idx]}"
  log_name="${TRAIN_LOG_NAMES[$idx]}"
  stage_name="${TRAIN_STAGE_NAMES[$idx]}"
  run_cmd "$log_name" "$stage_name" \
    python -m ship_motion.train --config "$RUNTIME_CONFIG_ROOT/$runtime_config"
done

PIPELINE_END_TS=$(date +%s)
print_banner "[Run Done] Follow-up Linux experiments finished | result_dir=$RESULT_DIR | total_elapsed=$(format_duration "$((PIPELINE_END_TS - PIPELINE_START_TS))")"
echo "[Next] 如需重汇总与画图，可执行："
echo "bash scripts/plot_formal_result.sh \"$PROJECT_ROOT\" \"$RESULT_NAME\""

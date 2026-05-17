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

run_cmd() {
  local name="$1"
  shift
  echo "=================================================="
  echo "RUNNING: $name"
  echo "=================================================="
  "$@" 2>&1 | tee "logs/${name}.log"
}

echo "==> Ensuring uv is available"
if ! command -v uv >/dev/null 2>&1; then
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "==> Syncing dependencies"
uv sync

echo "==> Step 03: build formal VMD cache"
run_cmd "step03_build_vmd_cache" \
  uv run python -m ship_motion.data.vmd --config configs/vmd_ccg_phys_xlstm.yaml

echo "==> Step 02/04/05/06: train main models"
run_cmd "train_lstm" \
  uv run python -m ship_motion.train --config configs/lstm.yaml

run_cmd "train_gru" \
  uv run python -m ship_motion.train --config configs/gru.yaml

run_cmd "train_transformer" \
  uv run python -m ship_motion.train --config configs/transformer.yaml

run_cmd "train_lite_xlstm" \
  uv run python -m ship_motion.train --config configs/lite_xlstm.yaml

run_cmd "train_ccg_xlstm" \
  uv run python -m ship_motion.train --config configs/ccg_xlstm.yaml

run_cmd "train_vmd_ccg_xlstm" \
  uv run python -m ship_motion.train --config configs/vmd_ccg_xlstm.yaml

run_cmd "train_vmd_ccg_phys_xlstm" \
  uv run python -m ship_motion.train --config configs/vmd_ccg_phys_xlstm.yaml

echo "==> Step 07: export predictions"
run_cmd "eval_lstm" \
  uv run python -m ship_motion.evaluate --config configs/lstm.yaml --checkpoint outputs/lstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_gru" \
  uv run python -m ship_motion.evaluate --config configs/gru.yaml --checkpoint outputs/gru_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_transformer" \
  uv run python -m ship_motion.evaluate --config configs/transformer.yaml --checkpoint outputs/transformer_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_lite_xlstm" \
  uv run python -m ship_motion.evaluate --config configs/lite_xlstm.yaml --checkpoint outputs/lite_xlstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_ccg_xlstm" \
  uv run python -m ship_motion.evaluate --config configs/ccg_xlstm.yaml --checkpoint outputs/ccg_xlstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_vmd_ccg_xlstm" \
  uv run python -m ship_motion.evaluate --config configs/vmd_ccg_xlstm.yaml --checkpoint outputs/vmd_ccg_xlstm_seq128_pred10/best.pt --split all --save-predictions

run_cmd "eval_vmd_ccg_phys_xlstm" \
  uv run python -m ship_motion.evaluate --config configs/vmd_ccg_phys_xlstm.yaml --checkpoint outputs/vmd_ccg_phys_xlstm_seq128_pred10/best.pt --split all --save-predictions

echo "==> Step 07: summarize and plot"
run_cmd "summarize_results" \
  uv run python -m ship_motion.summarize_results --ablation-config configs/ablation_list.yaml --run-output-root outputs --summary-output-dir outputs/summary

run_cmd "plot_results" \
  uv run python -m ship_motion.plot_results --ablation-config configs/ablation_list.yaml --run-output-root outputs --summary-output-dir outputs/summary

echo "==> Full training pipeline finished"

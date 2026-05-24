#!/usr/bin/env bash
set -euo pipefail

# 兼容旧入口：现在正式训练统一转发到 results/<result_name>/ 方案。
# 用法保持兼容：
#   bash scripts/run_full_training.sh . result_2

PROJECT_ROOT="${1:-$PWD}"
RESULT_NAME="${2:-result_2}"

exec bash "$PROJECT_ROOT/scripts/run_formal_result.sh" "$PROJECT_ROOT" "$RESULT_NAME"

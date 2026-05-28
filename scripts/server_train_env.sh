#!/usr/bin/env bash

# 双卡 24GB GPU / 24 核 CPU / 64GB 内存服务器的训练环境默认值。
# 说明：
# 1. 这里只提供“硬件提示”和线程上限；
# 2. 真正的 batch size / num_workers / precision 仍由训练代码按模型类型自动解析；
# 3. 如需临时覆盖，可在调用脚本前先自行 export 同名环境变量。

export SHIP_MOTION_HW_PROFILE="${SHIP_MOTION_HW_PROFILE:-dual_gpu_24g_24cpu_64g}"
export SHIP_MOTION_GPU_MEMORY_GB="${SHIP_MOTION_GPU_MEMORY_GB:-24}"
export SHIP_MOTION_CPU_CORES="${SHIP_MOTION_CPU_CORES:-24}"
export SHIP_MOTION_SYSTEM_MEMORY_GB="${SHIP_MOTION_SYSTEM_MEMORY_GB:-64}"
export SHIP_MOTION_PRECISION="${SHIP_MOTION_PRECISION:-auto}"

# 限制每个进程内部的 CPU 线程数，避免在双卡并发跑多个实验时过度抢核。
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

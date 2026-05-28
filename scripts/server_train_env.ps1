<#
双卡 24GB GPU / 24 核 CPU / 64GB 内存服务器的训练环境默认值。

说明：
1. 这里只设置硬件提示和线程上限；
2. 真正的 batch size / num_workers / precision 仍由训练代码按模型类型自动解析；
3. 如果外部已经显式设置同名环境变量，这里不会覆盖。
#>

if (-not $env:SHIP_MOTION_HW_PROFILE) { $env:SHIP_MOTION_HW_PROFILE = "dual_gpu_24g_24cpu_64g" }
if (-not $env:SHIP_MOTION_GPU_MEMORY_GB) { $env:SHIP_MOTION_GPU_MEMORY_GB = "24" }
if (-not $env:SHIP_MOTION_CPU_CORES) { $env:SHIP_MOTION_CPU_CORES = "24" }
if (-not $env:SHIP_MOTION_SYSTEM_MEMORY_GB) { $env:SHIP_MOTION_SYSTEM_MEMORY_GB = "64" }
if (-not $env:SHIP_MOTION_PRECISION) { $env:SHIP_MOTION_PRECISION = "auto" }

# 限制 BLAS / OpenMP 线程，避免在双卡并发跑多个实验时过度抢占 CPU。
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "4" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "4" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "4" }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = "4" }

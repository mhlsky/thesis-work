<#
单卡 40GB GPU / 12 核 CPU / 32GB 内存服务器的训练环境默认值。

说明：
1. 这里只设置硬件提示和线程上限；
2. 真正的 batch size / num_workers / precision 仍由训练代码按模型类型自动解析；
3. 如果外部已经显式设置同名环境变量，这里不会覆盖。
#>

if (-not $env:SHIP_MOTION_HW_PROFILE) { $env:SHIP_MOTION_HW_PROFILE = "single_gpu_40g_12cpu_32g" }
if (-not $env:SHIP_MOTION_GPU_MEMORY_GB) { $env:SHIP_MOTION_GPU_MEMORY_GB = "40" }
if (-not $env:SHIP_MOTION_CPU_CORES) { $env:SHIP_MOTION_CPU_CORES = "12" }
if (-not $env:SHIP_MOTION_SYSTEM_MEMORY_GB) { $env:SHIP_MOTION_SYSTEM_MEMORY_GB = "32" }
if (-not $env:SHIP_MOTION_PRECISION) { $env:SHIP_MOTION_PRECISION = "auto" }

# 限制 BLAS / OpenMP 线程，避免和 DataLoader worker 叠加后过度抢占 12 核 CPU。
if (-not $env:OMP_NUM_THREADS) { $env:OMP_NUM_THREADS = "6" }
if (-not $env:MKL_NUM_THREADS) { $env:MKL_NUM_THREADS = "6" }
if (-not $env:OPENBLAS_NUM_THREADS) { $env:OPENBLAS_NUM_THREADS = "6" }
if (-not $env:NUMEXPR_NUM_THREADS) { $env:NUMEXPR_NUM_THREADS = "6" }

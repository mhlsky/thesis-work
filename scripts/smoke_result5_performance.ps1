<#
实验5 smoke（Windows 便利入口）。

只加载已有的 LSTM / Joint checkpoint，并使用 CPU 的极少前向次数，
用于检查模型恢复、计时和 CSV/JSON 输出流程；不代表正式性能结果。
#>

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

Push-Location $repoRoot
try {
    $env:CUDA_VISIBLE_DEVICES = ""
    uv run python -m ship_motion.benchmark_inference `
        --config configs/experiment5_performance.yaml `
        --output-dir outputs/smoke/result5_performance `
        --smoke
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

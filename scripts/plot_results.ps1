<#
Step 07 结果汇总与绘图脚本。

默认行为：
1. 汇总各 run 的 metrics JSON；
2. 生成 routine/OOD 消融表和 physics 表；
3. 生成预测曲线图与柱状图。
#>

param(
    [switch]$Smoke,
    [string]$ResultName = ""
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if (-not $Smoke -and [string]::IsNullOrWhiteSpace($ResultName)) {
    $ResultName = "result_local"
    Write-Warning "未指定 -ResultName，正式汇总与绘图将默认写入 results/result_local/。"
}

$summaryDir = if ($Smoke) { "outputs/smoke/summary" } else { "results/$ResultName/outputs/summary" }
$runRoot = if ($Smoke) { "outputs/smoke" } else { "results/$ResultName/outputs" }

Push-Location $repoRoot
try {
    uv run python -m ship_motion.summarize_results --ablation-config configs/ablation_list.yaml --run-output-root $runRoot --summary-output-dir $summaryDir $(if ($Smoke) { "--smoke" })
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    uv run python -m ship_motion.plot_results --ablation-config configs/ablation_list.yaml --run-output-root $runRoot --summary-output-dir $summaryDir $(if ($Smoke) { "--smoke" })
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

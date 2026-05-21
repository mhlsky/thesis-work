<#
Step 07 smoke test。

目标：
1. 复用已有 smoke run；
2. 批量导出 predictions npz；
3. 验证结果汇总、物理指标表和绘图流程全部可跑通。
#>

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

Push-Location $repoRoot
try {
    Write-Host "==== Step 07: export smoke predictions ====" -ForegroundColor Cyan
    powershell -File .\scripts\evaluate_all.ps1 -Smoke
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "==== Step 07: summarize and plot smoke results ====" -ForegroundColor Cyan
    powershell -File .\scripts\plot_results.ps1 -Smoke
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

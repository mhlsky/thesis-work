<#
Lite-xLSTM 训练脚本。

默认使用正式配置训练；
如果加上 -Smoke，则切到极轻量流程，只用于本地快速验收。
#>

param(
    [switch]$Smoke
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$configPath = Join-Path $repoRoot "configs/lite_xlstm.yaml"
$startTime = Get-Date
$modeLabel = if ($Smoke) { "SMOKE" } else { "FULL" }

Write-Host ("=" * 80)
Write-Host "[Run Start] Stage=Train | Model=Lite-xLSTM | Mode=$modeLabel"
Write-Host "[Run Start] Config=$configPath"
Write-Host ("=" * 80)

Push-Location $repoRoot
try {
    if ($Smoke) {
        Write-Host "[Stage] Launch smoke training..."
        uv run python -m ship_motion.train --config $configPath --smoke
    }
    else {
        Write-Host "[Stage] Launch full training..."
        uv run python -m ship_motion.train --config $configPath
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $elapsed = (Get-Date) - $startTime
    $elapsedText = $elapsed.ToString("hh\:mm\:ss")
    Write-Host ("=" * 80)
    Write-Host "[Run Done] Model=Lite-xLSTM | TotalElapsed=$elapsedText"
    Write-Host ("=" * 80)
}
finally {
    Pop-Location
}

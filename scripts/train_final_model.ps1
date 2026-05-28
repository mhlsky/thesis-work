<#
最终主模型训练脚本：VMD-CCG-Phys-xLSTM。

默认使用正式配置训练；
如果加上 -Smoke，则先构建最小 VMD 缓存，再执行极轻量 smoke 训练。
#>

param(
    [switch]$Smoke,
    [string]$CacheRoot = "outputs/cache/vmd_smoke/K3_alpha2000"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
. (Join-Path $scriptDir "server_train_env.ps1")
$configPath = Join-Path $repoRoot "configs/vmd_ccg_phys_xlstm.yaml"
$startTime = Get-Date
$modeLabel = if ($Smoke) { "SMOKE" } else { "FULL" }

Write-Host ("=" * 80)
Write-Host "[Run Start] Stage=Train | Model=VMD-CCG-Phys-xLSTM | Mode=$modeLabel"
Write-Host "[Run Start] Config=$configPath"
if ($Smoke) {
    Write-Host "[Run Start] Smoke VMD cache root: $CacheRoot"
}
Write-Host ("=" * 80)

Push-Location $repoRoot
try {
    if ($Smoke) {
        Write-Host "[Stage] Build smoke VMD cache..."
        uv run python -m ship_motion.data.vmd --config $configPath --smoke --cache-root $CacheRoot
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
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
    Write-Host "[Run Done] Model=VMD-CCG-Phys-xLSTM | TotalElapsed=$elapsedText"
    Write-Host ("=" * 80)
}
finally {
    Pop-Location
}

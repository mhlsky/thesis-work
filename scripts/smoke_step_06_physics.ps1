<#
Step 06：物理约束与最终主模型 smoke test。

检查项：
1. 先构建最小 VMD smoke 缓存；
2. 验证最终模型能计算预测损失、VMD loss 和 raw 物理损失；
3. 验证评估阶段能输出 smoothness 与 roll_consistency_rmse。
#>

param(
    [string]$Config = "configs/vmd_ccg_phys_xlstm.yaml",
    [string]$CacheRoot = "outputs/cache/vmd_smoke/K3_alpha2000"
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

Push-Location $repoRoot
try {
    $configPath = Join-Path $repoRoot $Config
    if (-not (Test-Path -LiteralPath $configPath)) {
        Write-Error "Config not found: $configPath"
        exit 1
    }

    Write-Host "==== Step 06: build smoke VMD cache ====" -ForegroundColor Cyan
    uv run python -m ship_motion.data.vmd --config $configPath --smoke --cache-root $CacheRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "==== Step 06: smoke train final physical model ====" -ForegroundColor Cyan
    uv run python -m ship_motion.train --config $configPath --smoke
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

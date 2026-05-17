<#
Step 05：VMD-CCG-xLSTM smoke test。

检查项：
1. 先构建最小 VMD smoke 缓存；
2. 再验证 VMD-CCG-xLSTM 的“数据加载 -> 前向 -> 极小训练 -> 评估 -> 指标写出”流程；
3. 所有 smoke 输出默认写到 outputs/smoke/ 和 outputs/cache/vmd_smoke/。
#>

param(
    [string]$Config = "configs/vmd_ccg_xlstm.yaml",
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

    Write-Host "==== Step 05: build smoke VMD cache ====" -ForegroundColor Cyan
    uv run python -m ship_motion.data.vmd --config $configPath --smoke --cache-root $CacheRoot
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    Write-Host "==== Step 05: smoke train VMD-CCG-xLSTM ====" -ForegroundColor Cyan
    uv run python -m ship_motion.train --config $configPath --smoke
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

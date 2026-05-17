<#
Step 03 VMD 分解与标签缓存 smoke test。

检查项：
1. `vmdpy` 和 `numpy` 依赖是否可用；
2. train / validation / routine_test / ood_test 是否都能各自生成最小缓存；
3. Dataset 开启 VMD 后是否能返回 `y_modes`；
4. smoke 输出是否写入 `outputs/cache/vmd_smoke/`，避免污染正式缓存。
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

    Write-Host "==== Smoke testing Step 03 VMD cache ====" -ForegroundColor Cyan
    uv run python -m ship_motion.data.vmd --config $configPath --smoke --cache-root $CacheRoot
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

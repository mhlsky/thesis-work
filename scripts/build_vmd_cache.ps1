<#
为 Step 03 / Step 05 构建正式 VMD 标签缓存。

默认行为：
1. 读取 `configs/vmd_ccg_xlstm.yaml`；
2. 对 train / validation / routine_test / ood_test 四个划分逐个构建缓存；
3. 把缓存写到配置里的 `vmd.cache_root` 下。

如果你想临时把缓存写到其他地方，可以传入 `-CacheRoot`。
#>

param(
    [string]$Config = "configs/vmd_ccg_xlstm.yaml",
    [string]$CacheRoot = "",
    [string]$ResultName = ""
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

    if (-not [string]::IsNullOrWhiteSpace($ResultName)) {
        Write-Host "==== Preparing formal runtime config for result $ResultName ====" -ForegroundColor Yellow
        uv run python -m ship_motion.prepare_formal_result --result-name $ResultName --configs $Config --repo-root $repoRoot
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        $configPath = Join-Path $repoRoot ("results/{0}/runtime_configs/{1}" -f $ResultName, [System.IO.Path]::GetFileName($Config))
    }

    Write-Host "==== Building VMD cache with $Config ====" -ForegroundColor Cyan
    if (-not [string]::IsNullOrWhiteSpace($CacheRoot)) {
        uv run python -m ship_motion.data.vmd --config $configPath --cache-root $CacheRoot
    }
    else {
        uv run python -m ship_motion.data.vmd --config $configPath
    }
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

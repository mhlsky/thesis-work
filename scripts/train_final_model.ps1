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
$configPath = Join-Path $repoRoot "configs/vmd_ccg_phys_xlstm.yaml"

Push-Location $repoRoot
try {
    if ($Smoke) {
        uv run python -m ship_motion.data.vmd --config $configPath --smoke --cache-root $CacheRoot
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
        uv run python -m ship_motion.train --config $configPath --smoke
    }
    else {
        uv run python -m ship_motion.train --config $configPath
    }

    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

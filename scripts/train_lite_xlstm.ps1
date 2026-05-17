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

Push-Location $repoRoot
try {
    if ($Smoke) {
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

<#
Step 07 主线消融训练脚本。

默认顺序执行主线模型训练。
如果加上 -Smoke，则改为执行对应的 smoke 训练流程。
#>

param(
    [switch]$Smoke
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

$configs = @(
    "configs/persistence.yaml",
    "configs/lstm.yaml",
    "configs/gru.yaml",
    "configs/transformer.yaml",
    "configs/lite_xlstm.yaml",
    "configs/ccg_xlstm.yaml",
    "configs/vmd_ccg_xlstm.yaml",
    "configs/vmd_ccg_phys_xlstm.yaml"
)

Push-Location $repoRoot
try {
    foreach ($config in $configs) {
        $configPath = Join-Path $repoRoot $config
        if (-not (Test-Path -LiteralPath $configPath)) {
            Write-Error "Config not found: $configPath"
            exit 1
        }

        Write-Host "==== Training $config ====" -ForegroundColor Cyan
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
}
finally {
    Pop-Location
}

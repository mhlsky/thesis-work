<#
Step 07 批量评估与预测导出脚本。

作用：
1. 读取主线 run 的 checkpoint；
2. 重新导出 routine_test / ood_test 的 metrics 与 predictions；
3. 便于 Step 07 汇总与绘图直接消费。
#>

param(
    [switch]$Smoke
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

$runs = @(
    @{ Config = "configs/persistence.yaml"; Run = "persistence_seq128_pred10" },
    @{ Config = "configs/lstm.yaml"; Run = "lstm_seq128_pred10" },
    @{ Config = "configs/gru.yaml"; Run = "gru_seq128_pred10" },
    @{ Config = "configs/transformer.yaml"; Run = "transformer_seq128_pred10" },
    @{ Config = "configs/lite_xlstm.yaml"; Run = "lite_xlstm_seq128_pred10" },
    @{ Config = "configs/ccg_xlstm.yaml"; Run = "ccg_xlstm_seq128_pred10" },
    @{ Config = "configs/vmd_ccg_xlstm.yaml"; Run = "vmd_ccg_xlstm_seq128_pred10" },
    @{ Config = "configs/vmd_ccg_phys_xlstm.yaml"; Run = "vmd_ccg_phys_xlstm_seq128_pred10" }
)

Push-Location $repoRoot
try {
    foreach ($item in $runs) {
        $configPath = Join-Path $repoRoot $item.Config
        $runName = if ($Smoke) { "$($item.Run)_smoke" } else { $item.Run }
        $runRoot = if ($Smoke) { "outputs/smoke" } else { "outputs" }
        $checkpointPath = Join-Path $repoRoot "$runRoot/$runName/best.pt"

        if (-not (Test-Path -LiteralPath $configPath)) {
            Write-Error "Config not found: $configPath"
            exit 1
        }
        if (-not (Test-Path -LiteralPath $checkpointPath)) {
            Write-Error "Checkpoint not found: $checkpointPath"
            exit 1
        }

        Write-Host "==== Exporting predictions for $runName ====" -ForegroundColor Cyan
        if ($Smoke) {
            uv run python -m ship_motion.evaluate --config $configPath --checkpoint $checkpointPath --split all --smoke --save-predictions
        }
        else {
            uv run python -m ship_motion.evaluate --config $configPath --checkpoint $checkpointPath --split all --save-predictions
        }

        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

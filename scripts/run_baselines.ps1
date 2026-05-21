<#
顺序运行 Step 02 常规基线。

默认顺序：
1. Persistence
2. LSTM
3. GRU
4. TCN
5. Transformer

如果本地机器较慢，建议先手动只跑：
- LSTM
- GRU
- TCN

Transformer 可以后补，正式训练更建议放到算力平台执行。
#>

param(
    # 允许手动传入一组配置文件，便于只跑部分 baseline。
    [string[]]$Configs = @(
        "configs/persistence.yaml",
        "configs/lstm.yaml",
        "configs/gru.yaml",
        "configs/tcn.yaml",
        "configs/transformer.yaml"
    )
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

Push-Location $repoRoot
try {
    foreach ($config in $Configs) {
        Write-Host "==== Running $config ====" -ForegroundColor Cyan
        uv run python -m ship_motion.train --config $config
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

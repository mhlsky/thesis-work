<#
Step 04：Lite-xLSTM 与 CCG-xLSTM smoke test。

目标：
1. 验证 Lite-xLSTM 和 CCG-xLSTM 都能正常构建；
2. 验证“数据加载 -> 前向 -> 极小训练 -> 验证/测试评估 -> 输出文件写入”整条链路可跑通；
3. 默认把结果写到 outputs/smoke/，避免污染正式实验目录。
#>

param(
    # 默认顺序测试两个 Step 04 配置；
    # 如果你只想先测一个，可以通过命令行传入自定义数组。
    [string[]]$Configs = @(
        "configs/lite_xlstm.yaml",
        "configs/ccg_xlstm.yaml"
    )
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir

if (-not $Configs -or $Configs.Count -eq 0) {
    Write-Error "No configs provided."
    exit 1
}

Push-Location $repoRoot
try {
    foreach ($config in $Configs) {
        $configPath = Join-Path $repoRoot $config

        if (-not (Test-Path -LiteralPath $configPath)) {
            Write-Error "Config not found: $configPath"
            exit 1
        }

        Write-Host "==== Smoke testing $config ====" -ForegroundColor Cyan
        uv run python -m ship_motion.train --config $configPath --smoke
        if ($LASTEXITCODE -ne 0) {
            exit $LASTEXITCODE
        }
    }
}
finally {
    Pop-Location
}

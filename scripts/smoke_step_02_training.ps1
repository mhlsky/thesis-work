<#
Step 02 常规基线训练/评估 smoke test。

这个脚本的目标是：
1. 验证 uv 环境和依赖是否可用；
2. 验证“数据加载 -> 模型构建 -> 极小训练 -> 验证/测试评估 -> 指标文件写入”是否整条链路跑通；
3. 保证所有临时输出默认写到 outputs/smoke/ 下，而不是污染正式实验目录。

默认会顺序执行 5 个 baseline 的 smoke test：
1. Persistence
2. LSTM
3. GRU
4. TCN
5. Transformer

如果你只想跑其中几个模型：
- 可以在下面的默认数组里直接注释掉某些项；
- 也可以通过命令行传入自定义 `-Configs` 数组。
#>

param(
    # 允许一次传入一组配置文件，便于批量跑 smoke test。
    # 如果你只想临时关闭某个模型，可以直接把对应行注释掉。
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

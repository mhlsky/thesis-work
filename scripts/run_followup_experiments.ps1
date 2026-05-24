<#
第一次正式实验后的“补充实验”脚本。

设计目标：
1. 只跑“第一次实验后真正受改动影响”的模型；
2. 不重复训练 LSTM / GRU / Transformer / Lite-xLSTM / CCG-xLSTM 这些未改动主干；
3. 把新结果统一写到 results/<result_name>/ 下，方便和 result_1 对照。

为什么默认只跑这几个：
- persistence：result_1 里缺正式结果，需要补齐主表基线；
- vmd_ccg_xlstm：VMD warmup 改动会影响训练行为；
- vmd_ccg_phys_xlstm：VMD warmup + physics 约束实现/度量改动都会影响结果。

注意：
- 这个脚本面向“正式补充实验”，不是 smoke；
- train.py 在训练结束后会自动执行 val / routine_test / ood_test 评估并保存 predictions，
  所以这里不需要再额外调一次 evaluate_all。
#>

param(
    [string]$ResultName = "result_2_followup",
    [switch]$SkipPersistence,
    [switch]$IncludeCcgRepeat
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$resultDir = Join-Path $repoRoot "results/$ResultName"
$outputRoot = Join-Path $resultDir "outputs"
$logRoot = Join-Path $resultDir "logs"
$runtimeConfigRoot = Join-Path $resultDir "runtime_configs"

# 默认只纳入“第一次实验后需要补充验证”的模型。
$items = @()
if (-not $SkipPersistence) {
    $items += @{ Config = "configs/persistence.yaml"; Runtime = "persistence.yaml"; Stage = "followup_baseline"; Log = "train_persistence" }
}
$items += @{ Config = "configs/vmd_ccg_xlstm.yaml"; Runtime = "vmd_ccg_xlstm.yaml"; Stage = "followup_vmd"; Log = "train_vmd_ccg_xlstm" }
$items += @{ Config = "configs/vmd_ccg_phys_xlstm.yaml"; Runtime = "vmd_ccg_phys_xlstm.yaml"; Stage = "followup_physics"; Log = "train_vmd_ccg_phys_xlstm" }

# 如果你后面还想再验证“当前最佳 CCG-xLSTM 的稳定性”，可以显式打开这个开关。
if ($IncludeCcgRepeat) {
    $items += @{ Config = "configs/ccg_xlstm.yaml"; Runtime = "ccg_xlstm.yaml"; Stage = "followup_repro"; Log = "train_ccg_xlstm_repeat" }
}

New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
New-Item -ItemType Directory -Force -Path $runtimeConfigRoot | Out-Null

function Write-Banner {
    param([string]$Message)
    Write-Host ("=" * 80)
    Write-Host $Message
    Write-Host ("=" * 80)
}

function Invoke-LoggedCommand {
    param(
        [string]$Name,
        [string]$Stage,
        [scriptblock]$ScriptBlock
    )

    $logPath = Join-Path $logRoot "$Name.log"
    $startTime = Get-Date
    Write-Banner "[Task Start] name=$Name | stage=$Stage | started_at=$($startTime.ToString('yyyy-MM-dd HH:mm:ss'))"

    & $ScriptBlock 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        exit $exitCode
    }

    $elapsed = (Get-Date) - $startTime
    Write-Banner "[Task Done] name=$Name | stage=$Stage | elapsed=$($elapsed.ToString('hh\:mm\:ss')) | log=$logPath"
}

Push-Location $repoRoot
try {
    Write-Banner "[Run Start] Follow-up experiments after result_1 | result_name=$ResultName"
    Write-Host "[Info] 本次默认不会重复训练未改动主干的 baseline/main 模型。" -ForegroundColor Yellow

    $selectedConfigs = @($items | ForEach-Object { $_.Config })
    Invoke-LoggedCommand -Name "prepare_runtime_configs" -Stage "prepare_configs" -ScriptBlock {
        uv run python -m ship_motion.prepare_formal_result --result-name $ResultName --repo-root $repoRoot --configs $selectedConfigs
    }

    # 只要本次要跑任何 VMD 模型，就先共享构建一次正式 VMD cache。
    $needsVmd = $items | Where-Object { $_.Runtime -like "vmd_*" }
    if ($needsVmd.Count -gt 0) {
        Invoke-LoggedCommand -Name "step03_build_vmd_cache" -Stage "followup_vmd_cache" -ScriptBlock {
            uv run python -m ship_motion.data.vmd --config (Join-Path $runtimeConfigRoot "vmd_ccg_phys_xlstm.yaml")
        }
    }

    foreach ($item in $items) {
        $runtimeConfigPath = Join-Path $runtimeConfigRoot $item.Runtime
        Invoke-LoggedCommand -Name $item.Log -Stage $item.Stage -ScriptBlock {
            uv run python -m ship_motion.train --config $runtimeConfigPath
        }
    }

    Write-Banner "[Run Done] Follow-up experiments finished | result_dir=$resultDir"
    Write-Host "[Next] 你可以继续执行：" -ForegroundColor Green
    Write-Host "powershell -File .\scripts\plot_results.ps1 -ResultName $ResultName"
}
finally {
    Pop-Location
}

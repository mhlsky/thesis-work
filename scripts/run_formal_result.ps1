<#
Windows 正式实验一键脚本。

目标：
1. 把正式训练、评估、汇总、画图统一写到 `results/<result_name>/`；
2. 自动生成 runtime config，避免手工修改 YAML；
3. 生成可复查的 `logs/*.log`。

说明：
- 这个脚本面向正式实验，不用于 smoke。
- smoke 仍然沿用 `outputs/smoke/`，避免和正式实验目录混在一起。
#>

param(
    [string]$ResultName = "result_local",
    [int]$NumWindows = 200,
    [int]$Horizon = 0
)

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$resultDir = Join-Path $repoRoot "results/$ResultName"
$outputRoot = Join-Path $resultDir "outputs"
$logRoot = Join-Path $resultDir "logs"
$runtimeConfigRoot = Join-Path $resultDir "runtime_configs"

$configs = @(
    @{ Config = "persistence.yaml"; Run = "persistence_seq128_pred10"; Stage = "train_baselines"; Log = "train_persistence" },
    @{ Config = "lstm.yaml"; Run = "lstm_seq128_pred10"; Stage = "train_baselines"; Log = "train_lstm" },
    @{ Config = "gru.yaml"; Run = "gru_seq128_pred10"; Stage = "train_baselines"; Log = "train_gru" },
    @{ Config = "transformer.yaml"; Run = "transformer_seq128_pred10"; Stage = "train_baselines"; Log = "train_transformer" },
    @{ Config = "lite_xlstm.yaml"; Run = "lite_xlstm_seq128_pred10"; Stage = "train_xlstm"; Log = "train_lite_xlstm" },
    @{ Config = "ccg_xlstm.yaml"; Run = "ccg_xlstm_seq128_pred10"; Stage = "train_xlstm"; Log = "train_ccg_xlstm" },
    @{ Config = "vmd_ccg_xlstm.yaml"; Run = "vmd_ccg_xlstm_seq128_pred10"; Stage = "train_vmd_xlstm"; Log = "train_vmd_ccg_xlstm" },
    @{ Config = "vmd_ccg_phys_xlstm.yaml"; Run = "vmd_ccg_phys_xlstm_seq128_pred10"; Stage = "train_final_model"; Log = "train_vmd_ccg_phys_xlstm" }
)

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
    Write-Banner "[Run Start] Formal Windows experiment pipeline | result_name=$ResultName"

    Invoke-LoggedCommand -Name "prepare_runtime_configs" -Stage "prepare_configs" -ScriptBlock {
        uv run python -m ship_motion.prepare_formal_result --result-name $ResultName --repo-root $repoRoot
    }

    Invoke-LoggedCommand -Name "env_check" -Stage "environment" -ScriptBlock {
        @'
import sys
import torch

print(f"[Env] python: {sys.executable}")
print(f"[Env] torch: {torch.__version__}")
print(f"[Env] torch_cuda: {torch.version.cuda}")
print(f"[Env] cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[Env] device_name: {torch.cuda.get_device_name(0)}")
'@ | uv run python -
    }

    Invoke-LoggedCommand -Name "step03_build_vmd_cache" -Stage "step03_vmd_cache" -ScriptBlock {
        uv run python -m ship_motion.data.vmd --config (Join-Path $runtimeConfigRoot "vmd_ccg_phys_xlstm.yaml")
    }

    foreach ($item in $configs) {
        $runtimeConfigPath = Join-Path $runtimeConfigRoot $item.Config
        Invoke-LoggedCommand -Name $item.Log -Stage $item.Stage -ScriptBlock {
            uv run python -m ship_motion.train --config $runtimeConfigPath
        }
    }

    foreach ($item in $configs) {
        $runtimeConfigPath = Join-Path $runtimeConfigRoot $item.Config
        $checkpointPath = Join-Path $outputRoot "$($item.Run)/best.pt"
        $evalLogName = $item.Log -replace '^train_', 'eval_'
        Invoke-LoggedCommand -Name $evalLogName -Stage "eval_predictions" -ScriptBlock {
            uv run python -m ship_motion.evaluate --config $runtimeConfigPath --checkpoint $checkpointPath --split all --save-predictions
        }
    }

    Invoke-LoggedCommand -Name "summarize_results" -Stage "step07_summary" -ScriptBlock {
        uv run python -m ship_motion.summarize_results --ablation-config configs/ablation_list.yaml --run-output-root $outputRoot --summary-output-dir (Join-Path $outputRoot "summary")
    }

    Invoke-LoggedCommand -Name "plot_results" -Stage "step07_plot" -ScriptBlock {
        uv run python -m ship_motion.plot_results --ablation-config configs/ablation_list.yaml --run-output-root $outputRoot --summary-output-dir (Join-Path $outputRoot "summary") --num-windows $NumWindows --horizon $Horizon
    }

    Write-Banner "[Run Done] Formal Windows experiment pipeline finished | result_dir=$resultDir"
}
finally {
    Pop-Location
}

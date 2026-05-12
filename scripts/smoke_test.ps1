<#
这个脚本的作用：快速运行一次“数据流程冒烟测试”。

主要检查：
1. Python 是否能找到；
2. 项目模块是否能导入；
3. 配置文件是否能读取；
4. 数据集切窗和标准化流程是否能跑通。
#>

param(
    [string]$Python = $env:PYTHON
)

if ([string]::IsNullOrWhiteSpace($Python)) {
    $localPython = Join-Path $env:LOCALAPPDATA "Programs\Python\Python311\python.exe"
    if (Test-Path -LiteralPath $localPython) {
        $Python = $localPython
    }
    else {
        $Python = "python"
    }
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptDir
$srcDir = Join-Path $repoRoot "src"
$configPath = Join-Path $repoRoot "configs\base.yaml"

$existingPythonPath = $env:PYTHONPATH
if ([string]::IsNullOrWhiteSpace($existingPythonPath)) {
    $env:PYTHONPATH = $srcDir
}
else {
    $env:PYTHONPATH = "$srcDir;$existingPythonPath"
}

& $Python -m ship_motion.data.dataset --config $configPath --smoke

exit $LASTEXITCODE

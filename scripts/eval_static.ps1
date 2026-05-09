param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..")
)

Set-Location $ProjectRoot
python -m src.eval_static

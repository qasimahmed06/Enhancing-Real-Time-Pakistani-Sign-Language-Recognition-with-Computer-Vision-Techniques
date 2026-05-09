param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..")
)

Set-Location $ProjectRoot
python -m src.data.sanity_check

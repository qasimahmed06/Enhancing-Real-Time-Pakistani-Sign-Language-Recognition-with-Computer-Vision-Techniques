param(
    [string]$ImagePath,
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..")
)

if (-not $ImagePath) {
    throw "Pass -ImagePath <path-to-image>"
}

Set-Location $ProjectRoot
python -m src.infer_static --image $ImagePath

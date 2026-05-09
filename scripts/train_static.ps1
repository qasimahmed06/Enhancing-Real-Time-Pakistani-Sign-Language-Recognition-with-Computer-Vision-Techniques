param(
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\.."),
    [int]$Epochs = 10,
    [int]$BatchSize = 16
)

Set-Location $ProjectRoot
python -m src.train_static --epochs $Epochs --batch-size $BatchSize

#!/usr/bin/env python3
param(
    [string]$Video = $null,
    [string]$Folder = $null,
    [int]$TopK = 3,
    [float]$Threshold = 0.0
)

Set-Location "c:\Users\qasim\Desktop\PSL Stuff\Pakistan-Sign-Language-Project"

$cmd = ".\.venv\Scripts\python.exe -m src.infer_dynamic --top-k $TopK --threshold $Threshold"

if ($Video) {
    $cmd += " --video `"$Video`""
}
if ($Folder) {
    $cmd += " --folder `"$Folder`""
}

Invoke-Expression $cmd

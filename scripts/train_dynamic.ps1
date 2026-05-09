#!/usr/bin/env python3
param(
    [int]$Epochs = 30,
    [int]$BatchSize = 8,
    [string]$Resume = $null
)

Set-Location "c:\Users\qasim\Desktop\PSL Stuff\Pakistan-Sign-Language-Project"

if ($Resume) {
    .\.venv\Scripts\python.exe -m src.train_dynamic --epochs $Epochs --batch-size $BatchSize --resume $Resume
} else {
    .\.venv\Scripts\python.exe -m src.train_dynamic --epochs $Epochs --batch-size $BatchSize
}

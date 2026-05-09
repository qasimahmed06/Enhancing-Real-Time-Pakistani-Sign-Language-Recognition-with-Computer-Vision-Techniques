#!/usr/bin/env python3
param(
    [string]$Checkpoint = "checkpoints/dynamic_best.pth"
)

Set-Location "c:\Users\qasim\Desktop\PSL Stuff\Pakistan-Sign-Language-Project"
.\.venv\Scripts\python.exe -m src.eval_dynamic --checkpoint $Checkpoint

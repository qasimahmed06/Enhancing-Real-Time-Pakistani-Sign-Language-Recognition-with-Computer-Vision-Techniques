# Pakistan Sign Language Project (PyTorch)

This repository contains the Python implementation for static PSL recognition first, with dynamic and webcam phases planned next.

## 1) Environment Setup (Windows)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

3. Verify GPU support:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2) Build Static Splits

```powershell
python -m src.data.split_dataset
```

This creates:
- splits/train.csv
- splits/val.csv
- splits/test.csv
- splits/class_to_idx.json
- splits/split_stats.json

## 3) Train Static Model

```powershell
python -m src.train_static --epochs 10 --batch-size 16
```

Or with PowerShell helper:

```powershell
.\scripts\train_static.ps1 -Epochs 10 -BatchSize 16
```

## 4) Evaluate Static Model

```powershell
python -m src.eval_static
```

Outputs are saved in reports/.

## 5) Run Inference

```powershell
python -m src.infer_static --image "C:\path\to\image.jpg"
```

Or with helper:

```powershell
.\scripts\infer_static.ps1 -ImagePath "C:\path\to\image.jpg"
```

## 6) Dynamic Pipeline (Videos)

### Split Dynamic Videos
```powershell
python -m src.data.split_dynamic_dataset
```

### Train Dynamic Model
```powershell
python -m src.train_dynamic --epochs 30 --batch-size 8
```

### Evaluate Dynamic Model
```powershell
python -m src.eval_dynamic
```

### Run Video Inference
```powershell
python -m src.infer_dynamic --video "path/to/video.mp4"
python -m src.infer_dynamic --folder "path/to/video/folder"
```

## Notes

- Dataset root is currently configured in src/config.py.
- Static classes: 36 alphabet signs (images only)
- Dynamic classes: 4 video-based gestures (2-Hay, Alifmad, Aray, Jeem)
- Checkpoints and model artifacts are git-ignored by default.
- See NEXT_STEPS.md for detailed training instructions


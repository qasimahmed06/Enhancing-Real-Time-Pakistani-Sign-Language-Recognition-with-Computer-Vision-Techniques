import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from src.config import PATHS, STATIC_CFG
from src.data.transforms import get_eval_transforms
from src.models.static_inception import build_static_inception
from src.utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for static PSL classifier")
    parser.add_argument("--checkpoint", type=Path, default=PATHS.checkpoints_dir / "best.pth")
    parser.add_argument("--class-map", type=Path, default=PATHS.split_dir / "class_to_idx.json")
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--folder", type=Path, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    return parser.parse_args()


def predict_one(model, image_path: Path, transform, idx_to_class, device, top_k: int, threshold: float):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs = torch.softmax(logits, dim=1).squeeze(0)

    top_probs, top_idxs = torch.topk(probs, k=min(top_k, probs.numel()))
    top_probs = top_probs.detach().cpu().tolist()
    top_idxs = top_idxs.detach().cpu().tolist()

    pred_idx = top_idxs[0]
    pred_prob = top_probs[0]
    pred_label = idx_to_class[pred_idx] if pred_prob >= threshold else "unknown"

    return {
        "image": str(image_path),
        "prediction": pred_label,
        "confidence": pred_prob,
        "top_k": [
            {"label": idx_to_class[idx], "prob": prob}
            for idx, prob in zip(top_idxs, top_probs)
        ],
    }


def main() -> None:
    args = parse_args()
    if args.image is None and args.folder is None:
        raise ValueError("Provide --image or --folder")

    class_to_idx = load_json(args.class_map)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_static_inception(num_classes=len(class_to_idx), freeze_backbone=False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    transform = get_eval_transforms(STATIC_CFG.image_size)

    rows = []
    if args.image is not None:
        rows.append(
            predict_one(model, args.image, transform, idx_to_class, device, args.top_k, args.threshold)
        )
    if args.folder is not None:
        for p in sorted(args.folder.rglob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                rows.append(
                    predict_one(model, p, transform, idx_to_class, device, args.top_k, args.threshold)
                )

    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

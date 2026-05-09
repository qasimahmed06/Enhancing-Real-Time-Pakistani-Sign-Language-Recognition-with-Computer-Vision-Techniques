import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PATHS, STATIC_CFG
from src.data.static_dataset import ManifestImageDataset
from src.data.transforms import get_eval_transforms
from src.metrics.confusion import save_confusion_matrix
from src.models.static_inception import build_static_inception
from src.utils.io import load_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate static PSL classifier")
    parser.add_argument("--test-csv", type=Path, default=PATHS.split_dir / "test.csv")
    parser.add_argument("--checkpoint", type=Path, default=PATHS.checkpoints_dir / "best.pth")
    parser.add_argument("--class-map", type=Path, default=PATHS.split_dir / "class_to_idx.json")
    parser.add_argument("--batch-size", type=int, default=STATIC_CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=STATIC_CFG.num_workers)
    parser.add_argument("--out-dir", type=Path, default=PATHS.reports_dir)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    class_to_idx = load_json(args.class_map)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ManifestImageDataset(args.test_csv, transform=get_eval_transforms(STATIC_CFG.image_size))
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_static_inception(num_classes=len(class_names), freeze_backbone=False, unfreeze_mixed7c=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(loader, desc="test", leave=False)):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": acc,
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }

    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(
        all_targets,
        all_preds,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    with (args.out_dir / "classification_report.txt").open("w", encoding="utf-8") as f:
        f.write(report)

    save_confusion_matrix(
        y_true=all_targets,
        y_pred=all_preds,
        class_names=class_names,
        out_path=args.out_dir / "confusion_matrix.png",
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

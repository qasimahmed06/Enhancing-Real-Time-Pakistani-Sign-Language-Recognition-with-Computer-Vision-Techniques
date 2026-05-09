import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DYNAMIC_CFG, PATHS
from src.data.dynamic_dataset import DynamicVideoDataset
from src.data.transforms import get_eval_transforms
from src.metrics.confusion import save_confusion_matrix
from src.models.dynamic_lstm import build_dynamic_lstm
from src.utils.io import ensure_dir, load_json


def load_checkpoint(checkpoint_path: str, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_index, (frames, labels) in enumerate(tqdm(loader, desc="eval", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break

            frames = frames.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda"):
                logits = model(frames)
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / max(len(loader.dataset), 1), acc, p, r, f1, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate dynamic LSTM model")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dynamic_best.pth")
    parser.add_argument("--batch-size", type=int, default=DYNAMIC_CFG.batch_size)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    test_dataset = DynamicVideoDataset(
        PATHS.split_dir / "test_dynamic.csv",
        transform=get_eval_transforms(size=DYNAMIC_CFG.image_size),
        num_frames=DYNAMIC_CFG.num_frames,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=DYNAMIC_CFG.num_workers,
        pin_memory=DYNAMIC_CFG.pin_memory,
    )

    class_to_idx = load_json(PATHS.split_dir / "dynamic_class_to_idx.json")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_dynamic_lstm(
        num_classes=DYNAMIC_CFG.num_classes,
        hidden_dim=DYNAMIC_CFG.lstm_hidden_dim,
        num_layers=DYNAMIC_CFG.lstm_num_layers,
        dropout=DYNAMIC_CFG.dropout,
        attention_dim=DYNAMIC_CFG.attention_dim,
    )
    model = load_checkpoint(args.checkpoint, model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    loss, acc, precision, recall, f1, all_preds, all_targets = evaluate(
        model, test_loader, criterion, device, args.max_batches
    )

    metrics = {
        "loss": loss,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    print(json.dumps(metrics, indent=2))

    ensure_dir(PATHS.reports_dir)
    with open(PATHS.reports_dir / "dynamic_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    ensure_dir(PATHS.reports_dir)
    with open(PATHS.reports_dir / "dynamic_classification_report.txt", "w") as f:
        labels = list(range(len(class_names)))
        report = classification_report(all_targets, all_preds, target_names=class_names, labels=labels, zero_division=0)
        f.write(report)

    cm = confusion_matrix(all_targets, all_preds, labels=list(range(len(class_names))))
    save_confusion_matrix(all_targets, all_preds, class_names, PATHS.reports_dir / "dynamic_confusion_matrix.png")

    print(f"✓ Metrics saved to {PATHS.reports_dir}/dynamic_metrics.json")
    print(f"✓ Report saved to {PATHS.reports_dir}/dynamic_classification_report.txt")
    print(f"✓ Confusion matrix saved to {PATHS.reports_dir}/dynamic_confusion_matrix.png")


if __name__ == "__main__":
    main()

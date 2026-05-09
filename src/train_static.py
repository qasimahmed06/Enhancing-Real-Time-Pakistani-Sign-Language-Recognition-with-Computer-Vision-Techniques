import math
import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import PATHS, STATIC_CFG
from src.data.static_dataset import ManifestImageDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.static_inception import build_static_inception
from src.utils.io import ensure_dir
from src.utils.seed import seed_everything


def set_learning_rate(optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    use_amp,
    base_learning_rate,
    total_steps,
    start_step,
    max_batches=None,
):
    model.train()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_loss = 0.0
    all_preds, all_targets = [], []

    global_step = start_step
    for batch_index, (images, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            if isinstance(outputs, tuple):
                logits, aux_logits = outputs
                loss = criterion(logits, labels) + 0.4 * criterion(aux_logits, labels)
                preds = logits.argmax(dim=1)
            else:
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        progress = min(global_step, total_steps) / max(1, total_steps)
        learning_rate = 0.5 * base_learning_rate * (1.0 + math.cos(math.pi * progress))
        set_learning_rate(optimizer, learning_rate)

        total_loss += loss.item() * labels.size(0)
        all_preds.extend(preds.detach().cpu().tolist())
        all_targets.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), acc, p, r, f1


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch_index, (images, labels) in enumerate(tqdm(loader, desc="val", leave=False)):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(preds.detach().cpu().tolist())
            all_targets.extend(labels.detach().cpu().tolist())

    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), acc, p, r, f1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train static PSL classifier")
    parser.add_argument("--train-csv", type=Path, default=PATHS.split_dir / "train.csv")
    parser.add_argument("--val-csv", type=Path, default=PATHS.split_dir / "val.csv")
    parser.add_argument("--out-dir", type=Path, default=PATHS.checkpoints_dir)
    parser.add_argument("--epochs", type=int, default=STATIC_CFG.epochs)
    parser.add_argument("--batch-size", type=int, default=STATIC_CFG.batch_size)
    parser.add_argument("--num-workers", type=int, default=STATIC_CFG.num_workers)
    parser.add_argument("--lr", type=float, default=STATIC_CFG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=STATIC_CFG.weight_decay)
    parser.add_argument("--seed", type=int, default=STATIC_CFG.seed)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    train_ds = ManifestImageDataset(args.train_csv, transform=get_train_transforms(STATIC_CFG.image_size))
    val_ds = ManifestImageDataset(args.val_csv, transform=get_eval_transforms(STATIC_CFG.image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_static_inception(num_classes=STATIC_CFG.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=STATIC_CFG.label_smoothing)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = max(1, args.epochs * steps_per_epoch)

    start_epoch = 0
    best_f1 = -1.0
    global_step = start_epoch * steps_per_epoch

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", -1.0)
        global_step = ckpt.get("global_step", global_step)

    out_dir = ensure_dir(args.out_dir)
    history = []

    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            use_amp,
            args.lr,
            total_steps,
            global_step,
            max_batches=args.max_train_batches,
        )
        global_step += max(1, len(train_loader) if args.max_train_batches is None else min(args.max_train_batches, len(train_loader)))
        va_loss, va_acc, va_p, va_r, va_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            max_batches=args.max_val_batches,
        )

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_precision_macro": tr_p,
            "train_recall_macro": tr_r,
            "train_f1_macro": tr_f1,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_precision_macro": va_p,
            "val_recall_macro": va_r,
            "val_f1_macro": va_f1,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        last_payload = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_f1": best_f1,
            "global_step": global_step,
        }
        torch.save(last_payload, out_dir / "last.pth")

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_f1": best_f1,
                "global_step": global_step,
            }
            torch.save(best_payload, out_dir / "best.pth")

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"train_f1={tr_f1:.4f} val_f1={va_f1:.4f} "
            f"val_acc={va_acc:.4f}"
        )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

import argparse
import json
import math
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import DYNAMIC_CFG, PATHS
from src.data.dynamic_dataset import DynamicVideoDataset
from src.data.transforms import get_eval_transforms, get_train_transforms
from src.models.dynamic_lstm import build_dynamic_lstm
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
    for batch_index, (frames, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        if max_batches is not None and batch_index >= max_batches:
            break
        
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(frames)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

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
    return total_loss / len(loader.dataset), acc, p, r, f1


def load_checkpoint(checkpoint_path: str, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("best_f1", 0.0), checkpoint.get("global_step", 0)


def save_checkpoint(checkpoint_path: str, model, optimizer, epoch, best_f1, global_step):
    ensure_dir(Path(checkpoint_path).parent)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_f1": best_f1,
            "global_step": global_step,
        },
        checkpoint_path,
    )


def main():
    parser = argparse.ArgumentParser(description="Train dynamic LSTM model for video gestures")
    parser.add_argument("--epochs", type=int, default=DYNAMIC_CFG.epochs)
    parser.add_argument("--batch-size", type=int, default=DYNAMIC_CFG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DYNAMIC_CFG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DYNAMIC_CFG.weight_decay)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    seed_everything(DYNAMIC_CFG.seed)
    device = torch.device(args.device)

    train_dataset = DynamicVideoDataset(
        PATHS.split_dir / "train_dynamic.csv",
        transform=get_train_transforms(size=DYNAMIC_CFG.image_size),
        num_frames=DYNAMIC_CFG.num_frames,
    )
    val_dataset = DynamicVideoDataset(
        PATHS.split_dir / "val_dynamic.csv",
        transform=get_eval_transforms(size=DYNAMIC_CFG.image_size),
        num_frames=DYNAMIC_CFG.num_frames,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=DYNAMIC_CFG.num_workers,
        pin_memory=DYNAMIC_CFG.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=DYNAMIC_CFG.num_workers,
        pin_memory=DYNAMIC_CFG.pin_memory,
    )

    model = build_dynamic_lstm(
        num_classes=DYNAMIC_CFG.num_classes,
        hidden_dim=DYNAMIC_CFG.lstm_hidden_dim,
        num_layers=DYNAMIC_CFG.lstm_num_layers,
        dropout=DYNAMIC_CFG.dropout,
        attention_dim=DYNAMIC_CFG.attention_dim,
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=DYNAMIC_CFG.label_smoothing)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    best_f1 = 0.0
    global_step = 0

    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_epoch, best_f1, global_step = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1

    history = {"train_loss": [], "train_acc": [], "train_p": [], "train_r": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_p": [], "val_r": [], "val_f1": [], "lr": []}

    total_steps = len(train_loader) * args.epochs
    use_amp = args.device == "cuda"

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_p, train_r, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp,
            args.learning_rate, total_steps, global_step, args.max_train_batches
        )
        global_step += len(train_loader)

        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, criterion, device, args.max_val_batches)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_p"].append(train_p)
        history["train_r"].append(train_r)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_p"].append(val_p)
        history["val_r"].append(val_r)
        history["val_f1"].append(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"Learning Rate: {current_lr:.2e}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(str(PATHS.checkpoints_dir / "dynamic_best.pth"), model, optimizer, epoch, best_f1, global_step)
            print(f"✓ Best checkpoint saved (F1: {best_f1:.4f})")

        save_checkpoint(str(PATHS.checkpoints_dir / "dynamic_last.pth"), model, optimizer, epoch, best_f1, global_step)

    ensure_dir(PATHS.checkpoints_dir)
    with open(PATHS.checkpoints_dir / "dynamic_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()

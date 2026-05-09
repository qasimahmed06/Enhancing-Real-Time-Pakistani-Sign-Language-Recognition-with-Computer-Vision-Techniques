import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from src.config import PATHS, DYNAMIC_CFG
from src.utils.io import ensure_dir, save_json

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
DYNAMIC_CLASSES = {"2-Hay", "Alifmad", "Aray", "Jeem"}


def get_dynamic_class_folders(data_root: Path) -> list[Path]:
    folders = []
    for p in sorted(data_root.iterdir()):
        if p.is_dir() and p.name in DYNAMIC_CLASSES:
            folders.append(p)
    return folders


def get_videos(folder: Path) -> list[Path]:
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]


def split_items(items: list[Path], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    shuffled = items.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train : n_train + n_val]
    test = shuffled[n_train + n_val : n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filepath", "label", "label_idx"])
        writer.writeheader()
        writer.writerows(rows)


def build_dynamic_split(
    data_root: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    class_folders = get_dynamic_class_folders(data_root)
    if not class_folders:
        print("WARNING: No dynamic classes found!")
        return

    dynamic_classes = sorted([f.name for f in class_folders])
    class_to_idx = {label: idx for idx, label in enumerate(dynamic_classes)}

    split_rows = {"train": [], "val": [], "test": []}
    split_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for class_name in dynamic_classes:
        class_dir = data_root / class_name
        videos = get_videos(class_dir)
        grouped = split_items(videos, train_ratio, val_ratio, seed)
        for split_name, items in grouped.items():
            for item in items:
                split_rows[split_name].append(
                    {
                        "filepath": str(item),
                        "label": class_name,
                        "label_idx": class_to_idx[class_name],
                    }
                )
                split_counts[class_name][split_name] += 1

    for split_name in ["train", "val", "test"]:
        write_manifest(out_dir / f"{split_name}_dynamic.csv", split_rows[split_name])

    ensure_dir(out_dir)
    save_json(out_dir / "dynamic_class_to_idx.json", class_to_idx)
    save_json(out_dir / "split_stats_dynamic.json", dict(split_counts))

    print(f"Dynamic split complete:")
    print(f"  Classes: {dynamic_classes}")
    print(f"  Train: {len(split_rows['train'])} videos")
    print(f"  Val: {len(split_rows['val'])} videos")
    print(f"  Test: {len(split_rows['test'])} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dynamic classes into train/val/test")
    parser.add_argument("--data-root", type=Path, default=PATHS.data_root)
    parser.add_argument("--out-dir", type=Path, default=PATHS.split_dir)
    parser.add_argument("--train-ratio", type=float, default=DYNAMIC_CFG.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=DYNAMIC_CFG.val_ratio)
    parser.add_argument("--seed", type=int, default=DYNAMIC_CFG.seed)

    args = parser.parse_args()
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    build_dynamic_split(args.data_root, args.out_dir, args.train_ratio, args.val_ratio, test_ratio, args.seed)

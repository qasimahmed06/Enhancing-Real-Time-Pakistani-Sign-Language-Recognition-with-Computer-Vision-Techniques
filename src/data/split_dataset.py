import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from src.config import PATHS, STATIC_CFG
from src.utils.io import ensure_dir, save_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def get_class_folders(data_root: Path) -> list[Path]:
    return [p for p in sorted(data_root.iterdir()) if p.is_dir()]


def classify_folder_media(folder: Path) -> tuple[int, int]:
    images = 0
    videos = 0
    for file in folder.iterdir():
        if not file.is_file():
            continue
        ext = file.suffix.lower()
        if ext in IMAGE_EXTS:
            images += 1
        elif ext in VIDEO_EXTS:
            videos += 1
    return images, videos


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


def build_static_split(
    data_root: Path,
    out_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-8:
        raise ValueError("train/val/test ratios must sum to 1.0")

    class_folders = get_class_folders(data_root)
    static_classes: list[str] = []
    dynamic_classes: list[str] = []
    ignored_folders: list[str] = []

    for folder in class_folders:
        images, videos = classify_folder_media(folder)
        if images > 0 and videos == 0:
            static_classes.append(folder.name)
        elif videos > 0 and images == 0:
            dynamic_classes.append(folder.name)
        elif images == 0 and videos == 0:
            ignored_folders.append(folder.name)
        else:
            ignored_folders.append(folder.name)

    static_classes = sorted(static_classes)
    class_to_idx = {label: idx for idx, label in enumerate(static_classes)}

    split_rows = {"train": [], "val": [], "test": []}
    split_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    for class_name in static_classes:
        class_dir = data_root / class_name
        files = [
            p
            for p in sorted(class_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        grouped = split_items(files, train_ratio, val_ratio, seed)
        for split_name, items in grouped.items():
            for item in items:
                split_rows[split_name].append(
                    {
                        "filepath": str(item.resolve()),
                        "label": class_name,
                        "label_idx": str(class_to_idx[class_name]),
                    }
                )
            split_counts[class_name][split_name] = len(items)

    out_dir = ensure_dir(out_dir)
    write_manifest(out_dir / "train.csv", split_rows["train"])
    write_manifest(out_dir / "val.csv", split_rows["val"])
    write_manifest(out_dir / "test.csv", split_rows["test"])

    save_json(out_dir / "class_to_idx.json", class_to_idx)
    save_json(
        out_dir / "split_stats.json",
        {
            "seed": seed,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "num_static_classes": len(static_classes),
            "num_dynamic_classes": len(dynamic_classes),
            "dynamic_classes": dynamic_classes,
            "ignored_folders": ignored_folders,
            "per_class_counts": split_counts,
            "total_rows": {k: len(v) for k, v in split_rows.items()},
        },
    )

    train_set = {row["filepath"] for row in split_rows["train"]}
    val_set = {row["filepath"] for row in split_rows["val"]}
    test_set = {row["filepath"] for row in split_rows["test"]}
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Data leakage detected across splits")

    print(f"Static classes: {len(static_classes)}")
    print(f"Dynamic classes held out: {len(dynamic_classes)} -> {dynamic_classes}")
    print(f"Train rows: {len(split_rows['train'])}")
    print(f"Val rows: {len(split_rows['val'])}")
    print(f"Test rows: {len(split_rows['test'])}")
    print(f"Saved manifests in: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic static-only train/val/test manifests")
    parser.add_argument("--data-root", type=Path, default=PATHS.data_root)
    parser.add_argument("--out-dir", type=Path, default=PATHS.split_dir)
    parser.add_argument("--train-ratio", type=float, default=STATIC_CFG.train_ratio)
    parser.add_argument("--val-ratio", type=float, default=STATIC_CFG.val_ratio)
    parser.add_argument("--test-ratio", type=float, default=STATIC_CFG.test_ratio)
    parser.add_argument("--seed", type=int, default=STATIC_CFG.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_static_split(
        data_root=args.data_root,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

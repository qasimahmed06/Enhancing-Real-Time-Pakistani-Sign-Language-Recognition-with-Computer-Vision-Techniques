import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

from src.config import PATHS, STATIC_CFG
from src.data.static_dataset import ManifestImageDataset
from src.data.transforms import get_eval_transforms
from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a static data sanity check")
    parser.add_argument("--csv", type=Path, default=PATHS.split_dir / "train.csv")
    parser.add_argument("--out-path", type=Path, default=PATHS.reports_dir / "batch_preview.png")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = ManifestImageDataset(args.csv, transform=get_eval_transforms(STATIC_CFG.image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    images, labels = next(iter(loader))

    print(f"batch_shape={tuple(images.shape)}")
    print(f"labels={labels.tolist()}")
    print(f"label_min={int(labels.min().item())} label_max={int(labels.max().item())}")

    grid = make_grid(images, nrow=min(args.batch_size, 4), normalize=True, value_range=(-1, 1))
    ensure_dir(args.out_path.parent)
    plt.figure(figsize=(12, 8))
    plt.axis("off")
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.tight_layout()
    plt.savefig(args.out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"saved_preview={args.out_path}")


if __name__ == "__main__":
    main()

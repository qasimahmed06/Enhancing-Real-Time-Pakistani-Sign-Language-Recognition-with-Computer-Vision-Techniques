from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ManifestImageDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None):
        self.data = pd.read_csv(manifest_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        row = self.data.iloc[index]
        image_path = Path(row["filepath"])
        label_idx = int(row["label_idx"])

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label_idx

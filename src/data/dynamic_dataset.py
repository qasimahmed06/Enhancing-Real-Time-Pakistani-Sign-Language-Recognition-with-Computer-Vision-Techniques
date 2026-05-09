import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.config import DYNAMIC_CFG


class DynamicVideoDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None, num_frames: int = 30):
        self.manifest_path = Path(manifest_path)
        self.transform = transform
        self.num_frames = num_frames
        self.data = []
        self.labels = []

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row["filepath"])
                self.labels.append(int(row["label_idx"]))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path = self.data[idx]
        label = self.labels[idx]

        frames = self.extract_frames(video_path, self.num_frames)

        if self.transform:
            frames = [self.transform(f) for f in frames]
        else:
            frames = [torch.from_numpy(f).permute(2, 0, 1).float() / 255.0 for f in frames]

        frames = torch.stack(frames)
        return frames, label

    def extract_frames(self, video_path: str, num_frames: int) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            raise ValueError(f"Cannot read video: {video_path}")

        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (DYNAMIC_CFG.image_size, DYNAMIC_CFG.image_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                frames.append(np.zeros((DYNAMIC_CFG.image_size, DYNAMIC_CFG.image_size, 3), dtype=np.uint8))

        cap.release()
        return frames

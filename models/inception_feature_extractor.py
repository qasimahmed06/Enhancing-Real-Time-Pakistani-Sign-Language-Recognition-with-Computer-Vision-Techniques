"""InceptionV3-based feature extractor for dynamic PSL inference."""

from __future__ import annotations

from pathlib import Path

import torch
from torchvision.models import inception_v3
from torchvision.models.feature_extraction import create_feature_extractor

from models.checkpoint_utils import normalize_state_dict


class InceptionFeatureExtractor:
    def __init__(self, checkpoint_path=None, device=None, num_classes=36):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = inception_v3(weights=None, aux_logits=False, num_classes=num_classes)

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            state_dict = normalize_state_dict(checkpoint)
            self.model.load_state_dict(state_dict, strict=False)

        self.model = self.model.to(self.device).eval()
        self.feature_extractor = create_feature_extractor(self.model, return_nodes={"avgpool": "features"})

    @torch.no_grad()
    def extract_features(self, frames):
        squeeze_output = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_output = True

        batch_size, num_frames, C, H, W = frames.shape
        frames_flat = frames.view(-1, C, H, W).to(self.device)
        features = self.feature_extractor(frames_flat)["features"]
        features = torch.flatten(features, 1)
        features = features.view(batch_size, num_frames, -1)

        if squeeze_output:
            features = features.squeeze(0)

        return features


if __name__ == "__main__":
    extractor = InceptionFeatureExtractor()
    dummy_frames = torch.randn(10, 3, 224, 224)
    features = extractor.extract_features(dummy_frames)
    print(f"Features shape: {features.shape}")
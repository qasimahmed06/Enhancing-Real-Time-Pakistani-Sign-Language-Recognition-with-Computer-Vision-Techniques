import torch
import torch.nn as nn
import timm
from pathlib import Path

class VGGFeatureExtractor:
    def __init__(self, checkpoint_path=None, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.model = timm.create_model('vgg16', pretrained=True, num_classes=36)
        
        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model = self.model.to(self.device).eval()
        
        self.feature_extractor = nn.Sequential(
            self.model.features,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    @torch.no_grad()
    def extract_features(self, frames):
        squeeze_output = False
        if frames.dim() == 4:
            frames = frames.unsqueeze(0)
            squeeze_output = True
        
        batch_size, num_frames, C, H, W = frames.shape
        frames_flat = frames.view(-1, C, H, W).to(self.device)
        features = self.feature_extractor(frames_flat)
        features = features.view(batch_size, num_frames, -1)
        
        if squeeze_output:
            features = features.squeeze(0)
        
        return features
    
if __name__ == "__main__":
    extractor = VGGFeatureExtractor("checkpoints/vgg16_psl_best.pth")
    dummy_frames = torch.randn(10, 3, 224, 224)
    features = extractor.extract_features(dummy_frames)
    print(f"Features shape: {features.shape}")

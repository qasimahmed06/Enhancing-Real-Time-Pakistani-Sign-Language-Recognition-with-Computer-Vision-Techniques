"""Pre-extract VGG features from all videos and save to disk for fast LSTM training."""

import torch
import os
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vgg_feature_extractor import VGGFeatureExtractor
from dataloader.dataset_prep_videos import PSLVideoDataset, train_transform


def extract_and_save_features(dataset, vgg_extractor, save_dir='features', max_frames=30):
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Extracting features for {len(dataset)} videos...")
    
    for idx in tqdm(range(len(dataset))):
        frames, label, video_path = dataset[idx]
        
        with torch.no_grad():
            features = vgg_extractor.extract_features(frames)
        
        num_frames = features.shape[0]
        if num_frames < max_frames:
            padding = torch.zeros(max_frames - num_frames, 512, device=features.device)
            features = torch.cat([features, padding], dim=0)
        elif num_frames > max_frames:
            features = features[:max_frames]
        
        mask = torch.ones(max_frames, device=features.device)
        if num_frames < max_frames:
            mask[num_frames:] = 0
        
        video_name = os.path.basename(video_path).replace('.mp4', '.pt')
        save_path = os.path.join(save_dir, f"{label}_{idx}_{video_name}")
        
        torch.save({
            'features': features.cpu(),
            'label': label,
            'mask': mask.cpu(),
            'video_path': video_path
        }, save_path)
    
    print(f"✅ Saved features to {save_dir}/")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    vgg_extractor = VGGFeatureExtractor('checkpoints/vgg16_psl_best.pth', device=device)
    dataset = PSLVideoDataset('Dataset_Augmented', transform=train_transform, max_frames=30, sample_rate=2)
    extract_and_save_features(dataset, vgg_extractor, save_dir='features', max_frames=30)
    
    print("Done! Now you can use train_lstm_fast.py for much faster training.")
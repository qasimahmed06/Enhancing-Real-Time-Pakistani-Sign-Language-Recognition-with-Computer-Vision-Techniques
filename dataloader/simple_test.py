from torch.utils.data import DataLoader
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_prep import PSLDataset, train_transform as image_train_transform
from dataset_prep_videos import PSLVideoDataset, train_transform as video_train_transform
from models.vgg_feature_extractor import VGGFeatureExtractor
from models.lstm import PSL_LSTM

def test_pipeline():
    print("Testing complete pipeline...")
    
    # Video dataset
    dataset = PSLVideoDataset('Dataset', transform=video_train_transform, max_frames=30, sample_rate=2)
    print(f"Videos: {len(dataset)}, Classes: {len(dataset.label_map)}")
    
    if len(dataset) > 0:
        frames, label, video_path = dataset[0]
        print(f"Video shape: {frames.shape}, Label: {label}")
        
        # VGG features
        vgg_extractor = VGGFeatureExtractor('checkpoints/vgg16_psl_best.pth')
        features = vgg_extractor.extract_features(frames)
        print(f"VGG features: {features.shape}")
        
        # LSTM
        lstm_model = PSL_LSTM()
        features_batch = features.unsqueeze(0)
        mask = torch.ones(1, features.shape[0])
        
        with torch.no_grad():
            predictions = lstm_model(features_batch, mask=mask)
        
        predicted_class = predictions.argmax(dim=1)
        print(f"LSTM output: {predictions.shape}, Predicted: {predicted_class.item()}")

if __name__ == "__main__":
    test_pipeline()
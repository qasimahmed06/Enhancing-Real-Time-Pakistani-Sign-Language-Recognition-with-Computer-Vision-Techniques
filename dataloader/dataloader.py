from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import os

# Import both image and video datasets
from dataset_prep import PSLDataset, train_transform as image_train_transform, val_transform as image_val_transform
from dataset_prep_videos import PSLVideoDataset, train_transform as video_train_transform, val_transform as video_val_transform

# Import VGG feature extractor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vgg_feature_extractor import VGGFeatureExtractor
import torch


def test_image_dataset():
    """Test the image dataset (original)"""
    print("=" * 60)
    print("IMAGE DATASET TEST")
    print("=" * 60)
    
    dataset = PSLDataset(root_dir='Dataset', train_transform=image_train_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Total samples: {len(dataset)}")
    print(f"   Number of classes: {len(dataset.label_map)}")
    print(f"   Classes: {dataset.label_map}")
    
    # Get one batch
    for images, labels in loader:
        print(f"\n🔍 One Batch:")
        print(f"   Batch shape: {images.shape}")
        print(f"   └─ [batch_size, channels, height, width]")
        print(f"   └─ [{images.shape[0]}, {images.shape[1]}, {images.shape[2]}, {images.shape[3]}]")
        print(f"   Labels shape: {labels.shape}")
        print(f"   Labels: {labels.tolist()}")
        
        print(f"\n📸 One Training Example:")
        print(f"   Image shape: {images[0].shape}")
        print(f"   └─ [channels, height, width]")
        print(f"   └─ [{images[0].shape[0]}, {images[0].shape[1]}, {images[0].shape[2]}]")
        print(f"   Label: {labels[0].item()} ({dataset.label_map[labels[0].item()]})")
        print(f"   Pixel value range: [{images[0].min():.3f}, {images[0].max():.3f}]")
        break 
    
    print("\n" + "=" * 60)


def test_video_dataset():
    """Test the video dataset"""
    print("\n" + "=" * 60)
    print("VIDEO DATASET TEST")
    print("=" * 60)
    
    dataset = PSLVideoDataset(
        root_dir='Dataset', 
        transform=video_train_transform,
        max_frames=30,
        sample_rate=2
    )
    
    print(len(dataset))
    
    # Note: Can't use DataLoader with batch>1 for videos because 
    # different videos may have different number of frames
    # So we'll test with batch_size=1 or manually
    
    print(f"\n🔍 One Training Example (Video):")
    frames, label, video_path = dataset[0]
    
    print(f"   Video path: {video_path}")
    print(f"   Frames shape: {frames.shape}")
    print(f"   └─ [num_frames, channels, height, width]")
    print(f"   └─ [{frames.shape[0]}, {frames.shape[1]}, {frames.shape[2]}, {frames.shape[3]}]")
    print(f"   Label: {label} ({dataset.label_map[label]})")
    print(f"   Pixel value range: [{frames.min():.3f}, {frames.max():.3f}]")
    print(f"Classes: {dataset.label_map}")
    
    print(f"\n📹 Breaking it down:")
    print(f"   - This video has {frames.shape[0]} frames")
    print(f"   - Each frame is {frames.shape[1]} channels (RGB)")
    print(f"   - Each frame is {frames.shape[2]}x{frames.shape[3]} pixels")
    print(f"   - Total tensor size: {frames.numel():,} elements")
    
    print("\n" + "=" * 60)


def test_vgg_feature_extraction():
    """Test VGG feature extraction on video"""
    print("\n" + "=" * 60)
    print("VGG FEATURE EXTRACTION TEST")
    print("=" * 60)
    
    # Load video dataset
    dataset = PSLVideoDataset(
        root_dir='Dataset', 
        transform=video_train_transform,
        max_frames=30,
        sample_rate=2
    )
    
    # Load VGG feature extractor
    print(f"\n📦 Loading VGG16 feature extractor...")
    vgg_extractor = VGGFeatureExtractor(
        checkpoint_path='checkpoints/vgg16_psl_best.pth' if os.path.exists('checkpoints/vgg16_psl_best.pth') else None
    )
    
    # Get one video
    frames, label, video_path = dataset[0]
    
    print(f"\n🎬 Input Video:")
    print(f"   Video path: {video_path}")
    print(f"   Frames shape: {frames.shape}")
    print(f"   └─ [num_frames, channels, height, width]")
    print(f"   └─ [{frames.shape[0]}, {frames.shape[1]}, {frames.shape[2]}, {frames.shape[3]}]")
    
    # Extract features
    print(f"\n🔄 Extracting VGG features...")
    with torch.no_grad():
        features = vgg_extractor.extract_features(frames)
    
    print(f"\n✅ VGG Features Extracted:")
    print(f"   Features shape: {features.shape}")
    print(f"   └─ [num_frames, feature_dim]")
    print(f"   └─ [{features.shape[0]}, {features.shape[1]}]")
    print(f"   Feature value range: [{features.min():.3f}, {features.max():.3f}]")
    
    print(f"\n🧠 What this means:")
    print(f"   - Original: {frames.shape[0]} frames of {frames.shape[2]}x{frames.shape[3]}x{frames.shape[1]} = {frames.numel():,} values")
    print(f"   - Compressed: {features.shape[0]} frames of {features.shape[1]} features = {features.numel():,} values")
    print(f"   - Compression ratio: {frames.numel() / features.numel():.1f}x smaller")
    
    print(f"\n📊 LSTM Input:")
    print(f"   This is what goes into the LSTM:")
    print(f"   Shape: {features.shape}")
    print(f"   └─ [sequence_length, feature_dimension]")
    print(f"   └─ [{features.shape[0]} time steps, {features.shape[1]} features per step]")
    
    print("\n" + "=" * 60)


def show_full_pipeline():
    """Show the complete data flow"""
    print("\n" + "=" * 60)
    print("COMPLETE DATA PIPELINE")
    print("=" * 60)
    
    print(f"\n📝 Step-by-step breakdown:\n")
    
    print(f"1️⃣  VIDEO FILE (.mp4)")
    print(f"    └─ Contains multiple frames of sign language gesture")
    
    print(f"\n2️⃣  FRAME EXTRACTION (dataset_prep_videos.py)")
    print(f"    └─ Extract 30 frames from video")
    print(f"    └─ Apply transforms (resize, normalize, etc.)")
    print(f"    └─ Output: [30, 3, 224, 224]")
    print(f"              [frames, RGB, height, width]")
    
    print(f"\n3️⃣  VGG16 FEATURE EXTRACTION (vgg_feature_extractor.py)")
    print(f"    └─ Pass each frame through VGG16")
    print(f"    └─ Extract 512-dimensional feature vector per frame")
    print(f"    └─ Output: [30, 512]")
    print(f"              [frames, features]")
    
    print(f"\n4️⃣  LSTM MODEL (lstm.py)")
    print(f"    └─ Input: [batch_size, 30, 512]")
    print(f"    └─ LSTM processes sequence of features")
    print(f"    └─ Attention weighs important frames")
    print(f"    └─ Output: [batch_size, 4]")
    print(f"              [batch, num_classes]")
    
    print(f"\n5️⃣  PREDICTION")
    print(f"    └─ Softmax gives probability for each class")
    print(f"    └─ Argmax gives final prediction")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test PSL dataloaders')
    parser.add_argument('--test', type=str, default='all', 
                       choices=['images', 'videos', 'vgg', 'pipeline', 'all'],
                       help='Which test to run')
    args = parser.parse_args()
    
    if args.test in ['images', 'all']:
        test_image_dataset()
    
    if args.test in ['videos', 'all']:
        test_video_dataset()
    
    if args.test in ['vgg', 'all']:
        test_vgg_feature_extraction()
    
    if args.test in ['pipeline', 'all']:
        show_full_pipeline()
    
    print("\n✅ All tests complete!")
        

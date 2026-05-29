"""
Data Augmentation Script for PSL Videos
Creates deterministic augmented copies of videos to expand dataset size.
Each video gets multiple augmented versions with different transformations.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A
from pathlib import Path

# Define deterministic augmentation strategies (no randomness)
AUGMENTATION_STRATEGIES = {
    'original': A.Compose([
        A.Resize(224, 224)
    ]),
    
    'rotate_left': A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=(-10, -10), p=1.0)  # Always rotate -10 degrees
    ]),
    
    'rotate_right': A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=(10, 10), p=1.0)  # Always rotate +10 degrees
    ]),
    
    'brightness_up': A.Compose([
        A.Resize(224, 224),
        A.ColorJitter(brightness=(1.2, 1.2), contrast=1.0, saturation=1.0, hue=0, p=1.0)
    ]),
    
    'brightness_down': A.Compose([
        A.Resize(224, 224),
        A.ColorJitter(brightness=(0.8, 0.8), contrast=1.0, saturation=1.0, hue=0, p=1.0)
    ]),
    
    'contrast_up': A.Compose([
        A.Resize(224, 224),
        A.ColorJitter(brightness=1.0, contrast=(1.2, 1.2), saturation=1.0, hue=0, p=1.0)
    ]),
    
    'contrast_down': A.Compose([
        A.Resize(224, 224),
        A.ColorJitter(brightness=1.0, contrast=(0.8, 0.8), saturation=1.0, hue=0, p=1.0)
    ]),
    
    'crop_center': A.Compose([
        A.CenterCrop(height=200, width=200, p=1.0),
        A.Resize(224, 224)
    ]),
    
    'flip_horizontal': A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=1.0)  # Always flip (useful for symmetric signs)
    ]),
    
    'saturation_up': A.Compose([
        A.Resize(224, 224),
        A.ColorJitter(brightness=1.0, contrast=1.0, saturation=(1.3, 1.3), hue=0, p=1.0)
    ])
}


def augment_video(input_path, output_path, augmentation_transform, sample_rate=1):
    """
    Apply augmentation to a video and save the result.
    
    Args:
        input_path: Path to input video
        output_path: Path to save augmented video
        augmentation_transform: Albumentations transform to apply
        sample_rate: Sample every Nth frame
    """
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f"Failed to open video: {input_path}")
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (224, 224))
    
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames
        if frame_idx % sample_rate == 0:
            # Convert BGR to RGB for albumentations
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply augmentation
            augmented = augmentation_transform(image=frame_rgb)
            augmented_frame = augmented['image']
            
            # Convert back to BGR for video writer
            augmented_bgr = cv2.cvtColor(augmented_frame, cv2.COLOR_RGB2BGR)
            
            # Write frame
            out.write(augmented_bgr)
        
        frame_idx += 1
    
    cap.release()
    out.release()
    
    return True


def augment_dataset(
    input_dir='Dataset',
    output_dir='Dataset_Augmented',
    augmentations=None,
    sample_rate=1,
    skip_original=False
):
    """
    Create augmented versions of all videos in dataset.
    
    Args:
        input_dir: Directory containing class folders with videos
        output_dir: Directory to save augmented dataset
        augmentations: List of augmentation names to apply (None = all)
        sample_rate: Sample every Nth frame
        skip_original: If True, don't copy original videos
    """
    
    if augmentations is None:
        augmentations = list(AUGMENTATION_STRATEGIES.keys())
        if skip_original:
            augmentations.remove('original')
    
    print(f"Starting dataset augmentation...")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations: {augmentations}")
    print(f"Sample rate: {sample_rate}")
    
    # Statistics
    total_videos = 0
    total_generated = 0
    
    # Iterate through class folders
    for class_name in sorted(os.listdir(input_dir)):
        class_path = os.path.join(input_dir, class_name)
        
        if not os.path.isdir(class_path):
            continue
        
        # Get all video files
        video_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        if not video_files:
            continue
        
        print(f"\nProcessing class: {class_name} ({len(video_files)} videos)")
        
        # Create output class directory
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Process each video
        for video_file in tqdm(video_files, desc=f"{class_name}"):
            input_video_path = os.path.join(class_path, video_file)
            video_name = Path(video_file).stem
            
            total_videos += 1
            
            # Apply each augmentation
            for aug_name in augmentations:
                # Generate output filename
                output_filename = f"{video_name}_{aug_name}.mp4"
                output_video_path = os.path.join(output_class_dir, output_filename)
                
                # Apply augmentation
                success = augment_video(
                    input_video_path,
                    output_video_path,
                    AUGMENTATION_STRATEGIES[aug_name],
                    sample_rate=sample_rate
                )
                
                if success:
                    total_generated += 1
    
    print(f"\n{'='*60}")
    print(f"Augmentation Complete!")
    print(f"{'='*60}")
    print(f"Original videos: {total_videos}")
    print(f"Augmented videos: {total_generated}")
    print(f"Total videos: {total_videos + total_generated}")
    print(f"Multiplier: {(total_generated / total_videos):.1f}x")
    print(f"Output directory: {output_dir}")


def generate_custom_augmentations():
    """
    Generate only the most useful augmentations for sign language.
    Recommended: Use this for high-quality dataset expansion.
    """
    # Best augmentations for sign language (no horizontal flip as signs have handedness)
    recommended = [
        'original',           # Keep original
        'rotate_left',        # Slight rotation
        'rotate_right',       # Slight rotation
        'brightness_up',      # Lighting variation
        'brightness_down',    # Lighting variation
        'contrast_up',        # Better visibility
    ]
    
    return recommended


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Augment PSL video dataset')
    parser.add_argument('--input', type=str, default='Dataset', help='Input dataset directory')
    parser.add_argument('--output', type=str, default='Dataset_Augmented', help='Output directory')
    parser.add_argument('--sample-rate', type=int, default=1, help='Sample every Nth frame')
    parser.add_argument('--mode', type=str, default='recommended', 
                       choices=['all', 'recommended', 'minimal'],
                       help='Augmentation mode')
    
    args = parser.parse_args()
    
    # Choose augmentations based on mode
    if args.mode == 'all':
        augs = list(AUGMENTATION_STRATEGIES.keys())
        print("Mode: ALL augmentations (10x dataset)")
    elif args.mode == 'recommended':
        augs = generate_custom_augmentations()
        print("Mode: RECOMMENDED augmentations (6x dataset)")
    else:  # minimal
        augs = ['original', 'rotate_left', 'rotate_right', 'brightness_up']
        print("Mode: MINIMAL augmentations (4x dataset)")
    
    # Run augmentation
    augment_dataset(
        input_dir=args.input,
        output_dir=args.output,
        augmentations=augs,
        sample_rate=args.sample_rate,
        skip_original=False
    )
    
    print(f"\n✅ Done! You can now use '{args.output}' for training.")
    print(f"   Update your training script to use: root_dir='{args.output}'")

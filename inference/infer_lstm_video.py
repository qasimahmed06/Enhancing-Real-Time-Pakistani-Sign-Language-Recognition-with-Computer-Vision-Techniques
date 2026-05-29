#!/usr/bin/env python3
"""
FIXED webcam inference for LSTM PSL classifier with SLIDING WINDOW.
Matches training pipeline EXACTLY + optimal real-time performance.
"""

import argparse
import time
from pathlib import Path
import json
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import sys
from models.extract_features import VGGFeatureExtractor
from models.lstm import PSL_LSTM
from dataloader.dataset_prep_videos import val_transform
from dataloader.dataset_prep_videos import PSLVideoDataset


DEFAULT_LABEL_MAP = {
    0: '2-Hay', 
    1: 'Alifmad', 
    2: 'Aray', 
    3: 'Jeem'
}


def load_label_map_from_json(path):
    """Load label map from JSON file (list or dict format)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict):
            ordered = [data[str(i)] for i in range(len(data))]
            return ordered
        if isinstance(data, list):
            return data
    except Exception as e:
        print(f"Warning: could not load label map from JSON: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="FIXED LSTM PSL Classifier Webcam Inference (Sliding Window)")
    parser.add_argument('--lstm-checkpoint', type=str, default='checkpoints/lstm_psl_best.pth',
                        help='Path to LSTM checkpoint')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if not specified.')
    parser.add_argument('--label-map-json', type=str, default=None,
                        help='Optional JSON file with label map')
    parser.add_argument('--video-path', type=str, default=None,
                        help='Path to video file for single-video inference')
    parser.add_argument('--show-video', action='store_true',
                        help='Show video with prediction overlay (for --video-path)')
    parser.add_argument('--max-frames', type=int, default=30,
                        help='Maximum number of frames for LSTM sequence')
    parser.add_argument('--sample-rate', type=int, default=2,
                        help='Sample every Nth frame (e.g., 2 = sample every 2nd frame)')
    parser.add_argument('--prediction-smoothing', type=int, default=3,
                        help='Number of predictions to average for smoothing (1 = no smoothing)')
    parser.add_argument('--inference-interval', type=int, default=1,
                        help='Run inference every N sampled frames (1 = every frame, 5 = every 5th)')
    args = parser.parse_args()

    # Device - EXACT same logic
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"🖥️  Using device: {device}")

    # Load VGG extractor - EXACT same way as fixed script
    print("🔧 Loading VGG16 feature extractor...")
    extractor = VGGFeatureExtractor(device=device)
    print("✅ VGG16 feature extractor loaded!\n")

    # Load LSTM - EXACT same logic
    print("🔧 Loading LSTM model...")
    checkpoint = torch.load(args.lstm_checkpoint, map_location=device, weights_only=False)
    
    print("\n🔍 DEBUGGING CHECKPOINT:")
    print(f"Checkpoint keys: {checkpoint.keys()}")
    if 'val_acc' in checkpoint:
        print(f"Checkpoint Val Acc: {checkpoint['val_acc']:.2f}%")
    if 'epoch' in checkpoint:
        print(f"Checkpoint Epoch: {checkpoint['epoch']}")
    
    # Get num_classes from checkpoint
    state_dict = checkpoint['model_state_dict']
    classifier_weight_key = 'classifier.5.weight'
    
    if classifier_weight_key in state_dict:
        num_classes = state_dict[classifier_weight_key].shape[0]
        print(f"\n✅ CHECKPOINT has {num_classes} output classes")
    else:
        num_classes = len(DEFAULT_LABEL_MAP)
        print(f"\n⚠️  Could not detect num_classes from checkpoint, using default: {num_classes}")
    
    # Create model
    model = PSL_LSTM(input_size=512, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"✅ LSTM loaded with {num_classes} output classes")
    print(f"✅ Epoch {checkpoint.get('epoch', 'N/A')}, Val Acc: {checkpoint.get('val_acc', 0):.2f}%\n")

    # Load label map
    label_list = [DEFAULT_LABEL_MAP[i] for i in range(len(DEFAULT_LABEL_MAP))]
    
    if 'label_map' in checkpoint:
        lm = checkpoint['label_map']
        if isinstance(lm, dict):
            label_list = [lm[i] for i in range(len(lm))]
        elif isinstance(lm, list):
            label_list = lm
        print(f"📋 Loaded label_map from checkpoint: {len(label_list)} classes")
    elif args.label_map_json:
        lm = load_label_map_from_json(args.label_map_json)
        if lm:
            label_list = lm
            print(f"📋 Loaded label map from JSON: {len(label_list)} classes")
    
    # Ensure label_list matches num_classes
    if len(label_list) != num_classes:
        print(f"⚠️  Label list has {len(label_list)} classes but model has {num_classes} outputs")
        print(f"⚠️  Adjusting label list to match model...")
        if len(label_list) < num_classes:
            for i in range(len(label_list), num_classes):
                label_list.append(f"Class_{i}")
        else:
            label_list = label_list[:num_classes]
    
    print(f"📋 Final Classes: {label_list}\n")

    # --- VIDEO FILE OR WEBCAM MODE ---
    if args.video_path:
        print(f"📹 Opening video file: {args.video_path}...")
        cap = cv2.VideoCapture(args.video_path)
        source_name = f"Video: {Path(args.video_path).name}"
    else:
        print(f"📹 Opening camera {args.camera_id}...")
        cap = cv2.VideoCapture(args.camera_id)
        source_name = f"Camera {args.camera_id}"
    
    if not cap.isOpened():
        print(f'❌ Error: cannot open {source_name}')
        return

    print(f"✅ {source_name} opened successfully!")
    print(f"🎯 Mode: SLIDING WINDOW (rolling buffer of {args.max_frames} frames)")
    print(f"📊 Sample rate: every {args.sample_rate} frame(s)")
    print(f"🔄 Inference interval: every {args.inference_interval} sampled frame(s)")
    print(f"📈 Prediction smoothing: averaging last {args.prediction_smoothing} predictions")
    print("Press 'q' to quit\n")

    fps = 0.0
    last_time = 0.0
    
    # SLIDING WINDOW: Use deque for efficient rolling buffer
    feature_buffer = deque(maxlen=args.max_frames)
    
    # Prediction smoothing
    prediction_history = deque(maxlen=args.prediction_smoothing)
    
    frame_count = 0  # Total raw frame counter
    sampled_count = 0  # Sampled frame counter
    current_prediction = None
    current_confidence = 0.0
    predictions_made = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            frame_count += 1
            
            # Sample frames based on sample_rate
            if frame_count % args.sample_rate == 0:
                # Process full frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Preprocess frame using val_transform (EXACT same as training)
                transformed = val_transform(image=frame_rgb)
                frame_tensor = transformed['image']  # torch tensor, CxHxW
                
                # OPTIMIZATION: Extract features immediately for this single frame
                # This avoids re-running VGG on the entire history every time
                with torch.no_grad():
                    # Add batch dimension: (1, C, H, W)
                    single_frame_feat = extractor.extract_features(frame_tensor.unsqueeze(0).to(device))
                    # Output is (1, 512), take the first element to get (512,)
                    feature_vector = single_frame_feat[0]
                
                # Add to sliding window buffer
                feature_buffer.append(feature_vector)
                sampled_count += 1
                
                # Run inference when:
                # 1. Buffer has enough frames (at least min_frames)
                # 2. It's time to run inference based on inference_interval
                # Allow inference on partial buffers (e.g. start after 8 frames) to handle short videos
                min_frames_needed = min(args.max_frames, 8)
                
                if len(feature_buffer) >= min_frames_needed and sampled_count % args.inference_interval == 0:
                    with torch.no_grad():
                        # Stack features from buffer -> (N, 512)
                        features = torch.stack(list(feature_buffer)).to(device)
                        
                        # Prepare features for LSTM - EXACT same as FeatureDataset
                        num_frames, feature_dim = features.shape
                        
                        # Pad or truncate to max_frames (should already be correct)
                        if num_frames < args.max_frames:
                            padding = torch.zeros(args.max_frames - num_frames, feature_dim, device=features.device)
                            features = torch.cat([features, padding], dim=0)
                        elif num_frames > args.max_frames:
                            features = features[:args.max_frames]
                        
                        # Create mask
                        mask = torch.ones(args.max_frames, device=features.device)
                        if num_frames < args.max_frames:
                            mask[num_frames:] = 0
                        
                        # Add batch dimension
                        features = features.unsqueeze(0)  # (1, 30, 512)
                        mask = mask.unsqueeze(0)  # (1, 30)
                        
                        # Run LSTM with mask
                        outputs = model(features, mask=mask)
                        probs = F.softmax(outputs, dim=1)
                        
                        # Get prediction
                        top_prob, top_idx = torch.max(probs, dim=1)
                        top_idx = top_idx.item()
                        top_prob = top_prob.item()
                        
                        # Add to prediction history for smoothing
                        prediction_history.append((top_idx, top_prob))
                        predictions_made += 1
                        
                        # Smooth predictions: weighted vote with confidence
                        if args.prediction_smoothing > 1 and len(prediction_history) > 0:
                            # Count votes weighted by confidence
                            vote_scores = {}
                            for pred_idx, pred_conf in prediction_history:
                                if pred_idx not in vote_scores:
                                    vote_scores[pred_idx] = 0.0
                                vote_scores[pred_idx] += pred_conf
                            
                            # Get best voted prediction
                            smoothed_idx = max(vote_scores.items(), key=lambda x: x[1])[0]
                            smoothed_conf = vote_scores[smoothed_idx] / len(prediction_history)
                            
                            current_prediction = label_list[smoothed_idx] if smoothed_idx < len(label_list) else f"Class_{smoothed_idx}"
                            current_confidence = smoothed_conf
                        else:
                            # No smoothing
                            current_prediction = label_list[top_idx] if top_idx < len(label_list) else f"Class_{top_idx}"
                            current_confidence = top_prob

            # Display info
            buffer_size = len(feature_buffer)
            buffer_text = f"Buffer: {buffer_size}/{args.max_frames} frames"
            
            if current_prediction is not None:
                pred_text = f"{current_prediction}: {current_confidence*100:.1f}%"
                status_text = f"Predictions: {predictions_made}"
            else:
                pred_text = f"Warming up... ({buffer_size}/{args.max_frames})"
                status_text = "Collecting frames..."
            
            # Draw prediction with background
            (text_w, text_h), baseline = cv2.getTextSize(pred_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(display_frame, (10, 50), 
                         (20 + text_w, 60 + text_h + baseline), (0, 0, 0), -1)
            
            # Color based on confidence
            if current_prediction is not None:
                if current_confidence > 0.8:
                    color = (0, 255, 0)  # Green - high confidence
                elif current_confidence > 0.5:
                    color = (0, 255, 255)  # Yellow - medium confidence
                else:
                    color = (0, 165, 255)  # Orange - low confidence
            else:
                color = (200, 200, 200)  # Gray - warming up
            
            cv2.putText(display_frame, pred_text, (15, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            
            # Draw buffer status
            (buf_w, buf_h), buf_baseline = cv2.getTextSize(buffer_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(display_frame, (10, 90), 
                         (20 + buf_w, 100 + buf_h + buf_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, buffer_text, (15, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Draw status
            (st_w, st_h), st_baseline = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (10, 125), 
                         (20 + st_w, 135 + st_h + st_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, status_text, (15, 145), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Calculate and display FPS
            current_time = time.time()
            if last_time > 0:
                frame_time = current_time - last_time
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                fps = 0.9 * fps + 0.1 * current_fps
            last_time = current_time
            
            fps_text = f"FPS: {fps:.1f}"
            (fps_w, fps_h), fps_baseline = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display_frame, (5, 5), (15 + fps_w, 15 + fps_h + fps_baseline), (0, 0, 0), -1)
            cv2.putText(display_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 255, 50), 2)

            window_title = 'LSTM PSL Inference (Sliding Window) - Press q to quit'
            if args.video_path:
                window_title = f'Video: {Path(args.video_path).name} - Press q to quit'
            
            cv2.imshow(window_title, display_frame)
            
            # Run as fast as possible (1ms delay) to avoid slowing down inference
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Inference stopped")
        print(f"📊 Total predictions made: {predictions_made}")


if __name__ == '__main__':
    main()
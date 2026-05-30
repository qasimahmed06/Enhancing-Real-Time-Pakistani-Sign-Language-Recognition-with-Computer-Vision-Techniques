#!/usr/bin/env python3
"""
Combined Static (InceptionV3) and Dynamic (InceptionV3 + LSTM) Sign Language Recognition.

Automatically switches between static and dynamic models based on motion detection.
Static signs use an InceptionV3 classifier, dynamic gestures use InceptionV3 features with an LSTM attention head.

Usage:
    python infer_combined.py --video-path path/to/video.mp4
    python infer_combined.py  # Use webcam
"""

import argparse
import time
from pathlib import Path
import json
from collections import deque, Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions")
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from models.inception_feature_extractor import InceptionFeatureExtractor
from models.lstm import PSL_LSTM
from dataloader.dataset_prep_videos import val_transform
from models.checkpoint_utils import normalize_state_dict, infer_num_classes, infer_lstm_input_size
from roi_utils import center_crop_bbox, detect_hand_roi

class MotionDetector:
    def __init__(self, threshold=10, static_patience=5, dynamic_patience=2):
        self.prev_frame = None
        self.threshold = threshold
        self.static_patience = static_patience
        self.dynamic_patience = dynamic_patience
        
        self.consecutive_static = 0
        self.consecutive_dynamic = 0
        self.is_dynamic_state = False

    def update(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.prev_frame is None:
            self.prev_frame = gray
            return False, 0.0

        diff = cv2.absdiff(self.prev_frame, gray)
        self.prev_frame = gray
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        motion_score = np.sum(thresh) / 255.0
        motion_metric = motion_score / (gray.shape[0] * gray.shape[1]) * 1000

        is_frame_moving = motion_metric > self.threshold
        
        if is_frame_moving:
            self.consecutive_dynamic += 1
            self.consecutive_static = 0
        else:
            self.consecutive_static += 1
            self.consecutive_dynamic = 0

        if not self.is_dynamic_state:
            if self.consecutive_dynamic >= self.dynamic_patience:
                self.is_dynamic_state = True
        else:
            if self.consecutive_static >= self.static_patience:
                self.is_dynamic_state = False
                
        return self.is_dynamic_state, motion_metric

    def is_static(self):
        return not self.is_dynamic_state

    def is_dynamic(self):
        return self.is_dynamic_state
        
    def get_consecutive_static(self):
        return self.consecutive_static



def load_inception_model(checkpoint_path, device, num_classes=36):
    print(f"🔧 Loading InceptionV3 static model from {checkpoint_path}...")
    model = inception_v3(weights=None, aux_logits=False, num_classes=num_classes)
    
    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = normalize_state_dict(ckpt)
        inferred_classes = infer_num_classes(state_dict, default=num_classes)
        if inferred_classes != num_classes:
            model = inception_v3(weights=None, aux_logits=False, num_classes=inferred_classes)
            num_classes = inferred_classes
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"⚠️  Warning: missing keys: {len(missing)}")
        if unexpected:
            print(f"⚠️  Warning: unexpected keys: {len(unexpected)}")
    
    model = model.to(device).eval()
    return model

def load_lstm_model(checkpoint_path, device, input_size=2048):
    print(f"🔧 Loading LSTM dynamic model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = normalize_state_dict(checkpoint)
    num_classes = infer_num_classes(state_dict, default=4)
    input_size = infer_lstm_input_size(state_dict, default=input_size)
        
    model = PSL_LSTM(input_size=input_size, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    
    label_map = None
    if 'label_map' in checkpoint:
        lm = checkpoint['label_map']
        if isinstance(lm, dict):
            label_map = [lm[i] for i in range(len(lm))]
        elif isinstance(lm, list):
            label_map = lm
            
    return model, label_map, num_classes


class SentenceBuilder:
    def __init__(self):
        self.sentence = []
        self.max_length = 8
        self.static_history = deque(maxlen=5) 
        self.last_word = None
        self.cooldown = 0
        
    def update_static(self, label, conf):
        if conf < 0.6:
            return

        self.static_history.append(label)
        
        if len(self.static_history) == self.static_history.maxlen:
            counts = Counter(self.static_history)
            most_common, count = counts.most_common(1)[0]
            
            if count >= 4:
                if most_common != self.last_word:
                    self.sentence.append(most_common)
                    self.last_word = most_common
                    self.static_history.clear()
                    self._trim()
    
    def update_dynamic(self, label, conf):
        if conf < 0.5:
            return
        
        if label == self.last_word and self.cooldown > 0:
            return

        self.sentence.append(label)
        self.last_word = label
        self.static_history.clear()
        self.cooldown = 15
        self._trim()

    def step(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        
    def _trim(self):
        if len(self.sentence) > self.max_length:
            self.sentence.pop(0)
            
    def get_text(self):
        return " ".join(self.sentence)
    
    def clear(self):
        self.sentence = []
        self.last_word = None


def main():
    parser = argparse.ArgumentParser(description="Combined Static & Dynamic Gesture Inference")
    parser.add_argument('--video-path', type=str, default=None, help='Path to video file (default: webcam)')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID if no video path')
    parser.add_argument('--inception-checkpoint', '--vgg-checkpoint', dest='inception_checkpoint', type=str,
                        default='checkpoints/inceptionv3_psl_best.pth',
                        help='Path to the InceptionV3 static model checkpoint')
    parser.add_argument('--lstm-checkpoint', type=str, default='checkpoints/lstm_psl_best.pth')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--motion-threshold', type=float, default=0.5, help='Motion sensitivity threshold')
    parser.add_argument('--static-label-map', type=str, default=None, help='JSON for static labels')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")

    DEFAULT_STATIC_MAP = {
        0: '1-Hay', 1: 'Ain', 2: 'Alif', 3: 'Bay', 4: 'Byeh', 5: 'Chay', 6: 'Cyeh', 7: 'Daal',
        8: 'Dal', 9: 'Dochahay', 10: 'Fay', 11: 'Gaaf', 12: 'Ghain', 13: 'Hamza', 14: 'Kaf',
        15: 'Khay', 16: 'Kiaf', 17: 'Lam', 18: 'Meem', 19: 'Nuun', 20: 'Nuungh', 21: 'Pay',
        22: 'Ray', 23: 'Say', 24: 'Seen', 25: 'Sheen', 26: 'Suad', 27: 'Taay', 28: 'Tay',
        29: 'Tuey', 30: 'Wao', 31: 'Zaal', 32: 'Zaey', 33: 'Zay', 34: 'Zuad', 35: 'Zuey'
    }
    static_labels = [DEFAULT_STATIC_MAP[i] for i in range(len(DEFAULT_STATIC_MAP))]
    if args.static_label_map:
        with open(args.static_label_map, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                static_labels = [data[str(i)] for i in range(len(data))]
            else:
                static_labels = data
                
    vgg_model = load_inception_model(args.inception_checkpoint, device, num_classes=len(static_labels))
    
    lstm_model, dynamic_labels, num_dynamic_classes = load_lstm_model(args.lstm_checkpoint, device)
    if dynamic_labels is None:
        dynamic_labels = ['2-Hay', 'Alifmad', 'Aray', 'Jeem']
        if len(dynamic_labels) != num_dynamic_classes:
             dynamic_labels = [f"DynClass_{i}" for i in range(num_dynamic_classes)]
             
    print(f"📋 Static Classes: {len(static_labels)}")
    print(f"📋 Dynamic Classes: {len(dynamic_labels)} ({dynamic_labels})")
    
    feature_extractor = InceptionFeatureExtractor(device=device)

    hands = None
    if MEDIAPIPE_AVAILABLE:
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    else:
        print("⚠️  MediaPipe hands unavailable; combined mode will use OpenCV ROI fallback.")

    motion_detector = MotionDetector(threshold=args.motion_threshold, static_patience=5, dynamic_patience=3)
    sentence_builder = SentenceBuilder()

    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
        source_name = f"Video: {Path(args.video_path).name}"
    else:
        cap = cv2.VideoCapture(args.camera_id)
        source_name = f"Camera {args.camera_id}"
        
    if not cap.isOpened():
        print("❌ Error opening video source")
        return

    print(f"✅ Started inference on {source_name}")
    print("Press 'q' to quit")

    dynamic_feature_buffer = [] 
    dynamic_max_frames = 32
    seq_mode = False
    current_pred = "Waiting..."
    pred_conf = 0.0
    pred_type = "NONE"
    last_static_time = 0
    static_interval = 0.2

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        display_frame = frame.copy()
        
        is_moving, motion_val = motion_detector.update(frame)
        
        if motion_detector.is_static():
            state = "STATIC"
            color = (0, 255, 0)
        else:
            state = "DYNAMIC"
            color = (0, 0, 255)

        if state == "STATIC":
            if seq_mode:
                static_frames = motion_detector.get_consecutive_static()
                if static_frames > 0 and len(dynamic_feature_buffer) > static_frames:
                     dynamic_feature_buffer = dynamic_feature_buffer[:-static_frames]
                
                if len(dynamic_feature_buffer) >= 12:
                    print(f"⚡ Motion stopped. Running Dynamic Inference on {len(dynamic_feature_buffer)} frames...")
                    
                    features = torch.stack(dynamic_feature_buffer).to(device)
                    
                    with torch.no_grad():
                        target_len = 30
                        curr_len = features.shape[0]
                        if curr_len < target_len:
                            padding = torch.zeros(target_len - curr_len, 512, device=device)
                            features = torch.cat([features, padding], dim=0)
                        elif curr_len > target_len:
                            features = features[:target_len]
                            
                        mask = torch.ones(target_len, device=device)
                        if curr_len < target_len:
                            mask[curr_len:] = 0
                            
                        features = features.unsqueeze(0)
                        mask = mask.unsqueeze(0)
                        
                        out = lstm_model(features, mask=mask)
                        probs = F.softmax(out, dim=1)
                        top_prob, top_idx = torch.max(probs, dim=1)
                        
                        current_pred = dynamic_labels[top_idx.item()]
                        pred_conf = top_prob.item()
                        pred_type = "DYNAMIC"
                        
                        if pred_conf > 0.75:
                            sentence_builder.update_dynamic(current_pred, pred_conf)
                            print(f"   -> Prediction: {current_pred} ({pred_conf*100:.1f}%)")
                        else:
                            print(f"   -> Ignored low confidence: {current_pred} ({pred_conf*100:.1f}%)")
                
                dynamic_feature_buffer = []
                seq_mode = False
            
            if time.time() - last_static_time > static_interval:
                bbox, roi_source = detect_hand_roi(frame, mp_hands=hands, use_mediapipe=True, crop_ratio=0.80)
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

                    candidate_bboxes = [bbox, center_crop_bbox(frame, crop_ratio=0.80), (0, 0, frame.shape[1], frame.shape[0])]
                    candidate_probs = []

                    with torch.no_grad():
                        for cand_x, cand_y, cand_w, cand_h in candidate_bboxes:
                            roi = frame[cand_y:cand_y+cand_h, cand_x:cand_x+cand_w]
                            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                            transformed = val_transform(image=roi_rgb)
                            img_tensor = transformed['image'].unsqueeze(0).to(device)
                            out = vgg_model(img_tensor)
                            candidate_probs.append(F.softmax(out, dim=1))

                        probs = torch.stack(candidate_probs, dim=0).mean(dim=0)
                        top_prob, top_idx = torch.max(probs, dim=1)

                        current_pred = static_labels[top_idx.item()]
                        pred_conf = top_prob.item()
                        pred_type = "STATIC"

                        sentence_builder.update_static(current_pred, pred_conf)
                        
                last_static_time = time.time()

        elif state == "DYNAMIC":
            seq_mode = True
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            transformed = val_transform(image=frame_rgb)
            frame_tensor = transformed['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                feat = feature_extractor.extract_features(frame_tensor)
                feat = feat.squeeze(0)
            
            dynamic_feature_buffer.append(feat)
            
            if len(dynamic_feature_buffer) >= dynamic_max_frames:
                features = torch.stack(dynamic_feature_buffer).to(device)
                
                with torch.no_grad():
                    features = features[:30] 
                    
                    features = features.unsqueeze(0)
                    mask = torch.ones(1, 30, device=device)
                    
                    out = lstm_model(features, mask=mask)
                    probs = F.softmax(out, dim=1)
                    top_prob, top_idx = torch.max(probs, dim=1)
                    
                    current_pred = dynamic_labels[top_idx.item()]
                    pred_conf = top_prob.item()
                    pred_type = "DYNAMIC"
                    
                    sentence_builder.update_dynamic(current_pred, pred_conf)
                
                dynamic_feature_buffer = dynamic_feature_buffer[-15:]

        sentence_builder.step()

        cv2.rectangle(display_frame, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        cv2.putText(display_frame, f"STATE: {state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(display_frame, f"Motion: {motion_val:.1f}", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        pred_color = (0, 255, 255) if pred_type == "STATIC" else (255, 0, 255)
        cv2.putText(display_frame, f"PRED ({pred_type}): {current_pred} ({pred_conf*100:.1f}%)", 
                   (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
        
        if seq_mode:
            cv2.putText(display_frame, f"Buffer: {len(dynamic_feature_buffer)}", (frame.shape[1]-150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        sentence_text = sentence_builder.get_text()
        cv2.rectangle(display_frame, (0, frame.shape[0] - 60), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(display_frame, f"Sentence: {sentence_text}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow('Combined Inference', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

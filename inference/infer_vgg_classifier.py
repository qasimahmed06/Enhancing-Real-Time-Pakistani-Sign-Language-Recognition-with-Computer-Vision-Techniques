#!/usr/bin/env python3
"""
Inference script for the trained VGG16 classifier on prerecorded photos.

Behavior:
- Loads images from a folder (or a single image path)
- Applies the same validation preprocessing used during training (`val_transform`)
- Loads the VGG16 classifier checkpoint and runs inference
- Prints top-k predictions per image and can optionally display the image with overlay

Usage:
python infer_vgg_classifier.py --images-dir path/to/images --checkpoint checkpoints/vgg16_psl_best.pth

"""

import argparse
from pathlib import Path
import json
import time

import cv2
import torch
import torch.nn.functional as F
import timm
import mediapipe as mp

from dataloader.dataset_prep_videos import val_transform

# Default class mapping (from training). Index -> name
DEFAULT_LABEL_MAP = {
    0: '1-Hay', 1: 'Ain', 2: 'Alif', 3: 'Bay', 4: 'Byeh', 5: 'Chay', 6: 'Cyeh', 7: 'Daal',
    8: 'Dal', 9: 'Dochahay', 10: 'Fay', 11: 'Gaaf', 12: 'Ghain', 13: 'Hamza', 14: 'Kaf',
    15: 'Khay', 16: 'Kiaf', 17: 'Lam', 18: 'Meem', 19: 'Nuun', 20: 'Nuungh', 21: 'Pay',
    22: 'Ray', 23: 'Say', 24: 'Seen', 25: 'Sheen', 26: 'Suad', 27: 'Taay', 28: 'Tay',
    29: 'Tuey', 30: 'Wao', 31: 'Zaal', 32: 'Zaey', 33: 'Zay', 34: 'Zuad', 35: 'Zuey'
}


def load_label_map_from_json(path):
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


def build_model(checkpoint_path, device, num_classes=36, model_name='vgg16'):
    model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
            print("✅ Checkpoint loaded successfully!")
        else:
            try:
                model.load_state_dict(ckpt)
                print("✅ Checkpoint loaded successfully!")
            except Exception as e:
                print(f"⚠️  Warning: couldn't load checkpoint: {e}")
    model = model.to(device).eval()
    return model


def extract_hand_bbox(frame_bgr, mp_hands, results=None):
    """
    Extract hand bounding box using MediaPipe Hands detection.
    Returns: (bbox, hand_landmarks, handedness) or (None, None, None)
    bbox format: (x, y, w, h)
    """
    if results is None:
        # Convert BGR to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
    
    if not results.multi_hand_landmarks:
        return None, None, None
    
    H, W = frame_bgr.shape[:2]
    
    # If multiple hands detected, prefer right hand, otherwise use first
    selected_idx = 0
    selected_handedness = "Right"
    
    if results.multi_handedness and len(results.multi_handedness) > 1:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            label = hand_handedness.classification[0].label
            if label == "Right":
                selected_idx = idx
                selected_handedness = label
                break
    elif results.multi_handedness:
        selected_handedness = results.multi_handedness[0].classification[0].label
    
    hand_landmarks = results.multi_hand_landmarks[selected_idx]
    
    # Get bounding box from landmarks
    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    
    x_min = int(min(x_coords) * W)
    x_max = int(max(x_coords) * W)
    y_min = int(min(y_coords) * H)
    y_max = int(max(y_coords) * H)
    
    # Add padding (20% on each side)
    pad_x = int((x_max - x_min) * 0.20)
    pad_y = int((y_max - y_min) * 0.20)
    
    x_min = max(0, x_min - pad_x)
    y_min = max(0, y_min - pad_y)
    x_max = min(W, x_max + pad_x)
    y_max = min(H, y_max + pad_y)
    
    bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
    
    return bbox, hand_landmarks, selected_handedness


def gather_image_paths(images_dir_or_file):
    p = Path(images_dir_or_file)
    if p.is_file():
        return [p]
    elif p.is_dir():
        exts = ('.jpg', '.jpeg', '.png', '.bmp')
        files = sorted([x for x in p.iterdir() if x.suffix.lower() in exts])
        return files
    else:
        raise ValueError(f"No such file or directory: {images_dir_or_file}")


def main():
    parser = argparse.ArgumentParser(description="VGG16 Classifier Inference on Images")
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to a single image OR a directory containing images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/vgg16_psl_best.pth',
                        help='Path to VGG checkpoint')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/mps/cpu). Auto-detected if not specified.')
    parser.add_argument('--label-map-json', type=str, default=None,
                        help='Optional JSON file with label map')
    parser.add_argument('--topk', type=int, default=3,
                        help='Number of top predictions to print')
    parser.add_argument('--show', action='store_true', help='Show each image with prediction overlay')
    parser.add_argument('--delay', type=int, default=800, help='Display delay in ms when --show is used')
    args = parser.parse_args()

    # device
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

    # gather images
    image_paths = gather_image_paths(args.image_path)
    if len(image_paths) == 0:
        print("❌ No images found")
        return
    print(f"📷 Found {len(image_paths)} images")

    # label map
    label_list = [DEFAULT_LABEL_MAP[i] for i in range(len(DEFAULT_LABEL_MAP))]
    if args.label_map_json:
        lm = load_label_map_from_json(args.label_map_json)
        if lm:
            label_list = lm
            print(f"📋 Loaded label map from JSON: {len(label_list)} classes")

    if args.checkpoint and Path(args.checkpoint).exists():
        try:
            ckpt = torch.load(args.checkpoint, map_location='cpu')
            if isinstance(ckpt, dict) and 'label_map' in ckpt:
                lm = ckpt['label_map']
                if isinstance(lm, dict):
                    ordered = [lm[i] for i in range(len(lm))]
                    label_list = ordered
                elif isinstance(lm, list):
                    label_list = lm
                print(f"📋 Loaded label_map from checkpoint: {len(label_list)} classes")
        except Exception:
            pass

    # build model
    model = build_model(args.checkpoint, device, num_classes=len(label_list), model_name='vgg16')
    print(f"🎯 Model: VGG16 with {len(label_list)} classes")

    # Initialize MediaPipe Hands
    print("🤚 Initializing MediaPipe Hands...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,  # True for images
        max_num_hands=2,
        min_detection_confidence=0.5
    )

    # process images
    for idx, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Extract hand
        bbox, hand_landmarks, handedness = extract_hand_bbox(img_bgr, hands)
        
        if bbox is not None:
            x, y, w, h = bbox
            roi = img_bgr[y:y+h, x:x+w]
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            status_msg = f"Hand detected ({handedness})"
        else:
            # Fallback: use center crop or full image
            print(f"⚠️  No hand detected in {img_path.name}, using full image")
            roi_rgb = img_rgb
            bbox = None
            status_msg = "No hand detected"

        # use same validation transform
        transformed = val_transform(image=roi_rgb)
        img_tensor = transformed['image'].unsqueeze(0).to(device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)
            topk = torch.topk(probs, k=min(args.topk, probs.size(1)), dim=1)
            top_indices = topk.indices[0].tolist()
            top_probs = topk.values[0].tolist()
        t1 = time.time()

        # print results
        print(f"[{idx}/{len(image_paths)}] {img_path.name} - {status_msg} - Inference: {(t1-t0)*1000:.1f} ms")
        for rank, (i_cls, p) in enumerate(zip(top_indices, top_probs), 1):
            name = label_list[i_cls] if i_cls < len(label_list) else f"Class_{i_cls}"
            print(f"  {rank}. {name}: {p*100:.2f}%")

        if args.show:
            display = img_bgr.copy()
            
            # Draw bbox if found
            if bbox is not None:
                x, y, w, h = bbox
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            main_text = f"{label_list[top_indices[0]]}: {top_probs[0]*100:.1f}%"
            (w, h), baseline = cv2.getTextSize(main_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(display, (5, 5), (10 + w, 15 + h + baseline), (0, 0, 0), -1)
            cv2.putText(display, main_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow('VGG Classifier - press any key to continue', display)
            key = cv2.waitKey(args.delay)
            if key == 27:  # ESC to quit early
                break
    
    hands.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

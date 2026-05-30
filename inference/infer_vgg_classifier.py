#!/usr/bin/env python3
"""
Legacy implementation for the trained InceptionV3 classifier on prerecorded photos.

Behavior:
- Loads images from a folder (or a single image path)
- Applies the same validation preprocessing used during training (`val_transform`)
- Loads the InceptionV3 classifier checkpoint and runs inference
- Prints top-k predictions per image and can optionally display the image with overlay

Usage:
python infer_inception_classifier.py --image-path path/to/images --checkpoint checkpoints/inceptionv3_psl_best.pth

"""

import argparse
from pathlib import Path
import json
import time

import cv2
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = hasattr(mp, "solutions")
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

from dataloader.dataset_prep_videos import val_transform
from models.checkpoint_utils import normalize_state_dict, infer_num_classes
from roi_utils import center_crop_bbox, detect_hand_roi

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


def load_class_names_from_checkpoint(checkpoint_path):
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return None

    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            class_names = ckpt.get('class_names')
            if isinstance(class_names, dict):
                return [class_names[i] for i in range(len(class_names))]
            if isinstance(class_names, list):
                return class_names
    except Exception as exc:
        print(f"Warning: could not load class names from checkpoint: {exc}")
    return None


def build_model(checkpoint_path, device, num_classes=36, model_name='inception_v3'):
    model = inception_v3(weights=None, aux_logits=False, num_classes=num_classes)
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
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
        print("✅ Checkpoint loaded successfully!")
    model = model.to(device).eval()
    return model


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
    parser = argparse.ArgumentParser(description="InceptionV3 Classifier Inference on Images")
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to a single image OR a directory containing images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/inceptionv3_psl_best.pth',
                        help='Path to InceptionV3 checkpoint')
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
    else:
        checkpoint_labels = load_class_names_from_checkpoint(args.checkpoint)
        if checkpoint_labels:
            label_list = checkpoint_labels
            print(f"📋 Loaded class_names from checkpoint: {len(label_list)} classes")

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
    model = build_model(args.checkpoint, device, num_classes=len(label_list), model_name='inception_v3')
    print(f"🎯 Model: InceptionV3 with {len(label_list)} classes")

    # Initialize MediaPipe Hands when available; ROI helper will still fall back to skin/center crops.
    hands = None
    if MEDIAPIPE_AVAILABLE:
        print("🤚 Initializing MediaPipe Hands...")
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
    else:
        print("⚠️  MediaPipe hands unavailable; using OpenCV ROI fallback.")

    # process images
    for idx, img_path in enumerate(image_paths, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"❌ Could not read image: {img_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        bbox, roi_source = detect_hand_roi(img_bgr, mp_hands=hands, use_mediapipe=True, crop_ratio=0.80)
        t0 = time.time()
        with torch.no_grad():
            candidate_bboxes = [bbox, center_crop_bbox(img_bgr, crop_ratio=0.80), (0, 0, img_bgr.shape[1], img_bgr.shape[0])]
            candidate_sources = [roi_source, "center", "full"]
            candidate_probs = []

            for (x, y, w, h), source in zip(candidate_bboxes, candidate_sources):
                roi = img_bgr[y:y+h, x:x+w]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                transformed = val_transform(image=roi_rgb)
                img_tensor = transformed['image'].unsqueeze(0).to(device)
                outputs = model(img_tensor)
                probs = F.softmax(outputs, dim=1)
                candidate_probs.append(probs)

            probs = torch.stack(candidate_probs, dim=0).mean(dim=0)
            topk = torch.topk(probs, k=min(args.topk, probs.size(1)), dim=1)
            top_indices = topk.indices[0].tolist()
            top_probs = topk.values[0].tolist()
        t1 = time.time()

        # print results
        print(f"[{idx}/{len(image_paths)}] {img_path.name} - ROI ensemble ({roi_source}) - Inference: {(t1-t0)*1000:.1f} ms")
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
    
    if hands is not None:
        hands.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

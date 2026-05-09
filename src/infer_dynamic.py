import argparse
import json
from pathlib import Path

import torch

from src.config import DYNAMIC_CFG, PATHS
from src.data.dynamic_dataset import DynamicVideoDataset
from src.data.transforms import get_eval_transforms
from src.models.dynamic_lstm import build_dynamic_lstm
from src.utils.io import load_json


def load_checkpoint(checkpoint_path: str, model):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def infer_video(model, video_path: str, device, class_names: list[str], threshold: float = 0.0) -> dict:
    model.eval()
    
    dataset = DynamicVideoDataset(
        PATHS.split_dir / "test_dynamic.csv",
        transform=get_eval_transforms(size=DYNAMIC_CFG.image_size),
        num_frames=DYNAMIC_CFG.num_frames,
    )
    
    frames, _ = dataset.extract_frames(video_path, DYNAMIC_CFG.num_frames)
    
    if dataset.transform:
        frames = [dataset.transform(f) for f in frames]
    
    frames = torch.stack(frames).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)
    
    probs = probs.detach().cpu()[0]
    pred_idx = torch.argmax(probs).item()
    pred_prob = probs[pred_idx].item()
    
    if pred_prob < threshold:
        return {
            "video": str(video_path),
            "prediction": "unknown",
            "confidence": 0.0,
            "top_k": [],
        }
    
    top_k_probs, top_k_indices = torch.topk(probs, k=min(3, len(class_names)))
    top_k = [
        {"label": class_names[idx.item()], "prob": prob.item()}
        for idx, prob in zip(top_k_indices, top_k_probs)
    ]
    
    return {
        "video": str(video_path),
        "prediction": class_names[pred_idx],
        "confidence": pred_prob,
        "top_k": top_k,
    }


def main():
    parser = argparse.ArgumentParser(description="Inference on video files")
    parser.add_argument("--video", type=str, help="Path to single video file")
    parser.add_argument("--folder", type=str, help="Path to folder with video files")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/dynamic_best.pth")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)

    class_to_idx = load_json(PATHS.split_dir / "dynamic_class_to_idx.json")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_dynamic_lstm(
        num_classes=DYNAMIC_CFG.num_classes,
        hidden_dim=DYNAMIC_CFG.lstm_hidden_dim,
        num_layers=DYNAMIC_CFG.lstm_num_layers,
        dropout=DYNAMIC_CFG.dropout,
        attention_dim=DYNAMIC_CFG.attention_dim,
    )
    model = load_checkpoint(args.checkpoint, model)
    model = model.to(device)

    results = []

    if args.video:
        video_path = Path(args.video)
        if video_path.exists():
            print(f"Inferring on {video_path}...")
            result = infer_video(model, str(video_path), device, class_names, args.threshold)
            results.append(result)
            print(json.dumps(result, indent=2))
        else:
            print(f"Video file not found: {video_path}")

    elif args.folder:
        folder_path = Path(args.folder)
        if folder_path.is_dir():
            video_files = list(folder_path.glob("**/*.mp4")) + list(folder_path.glob("**/*.avi"))
            print(f"Found {len(video_files)} videos in {folder_path}")
            for video_path in video_files:
                print(f"Inferring on {video_path.name}...")
                result = infer_video(model, str(video_path), device, class_names, args.threshold)
                results.append(result)
            print(json.dumps(results, indent=2))
        else:
            print(f"Folder not found: {folder_path}")
    else:
        print("Please specify --video or --folder")


if __name__ == "__main__":
    main()

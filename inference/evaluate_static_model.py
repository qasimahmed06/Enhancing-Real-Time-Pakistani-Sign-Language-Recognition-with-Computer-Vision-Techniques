import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog


# ===========================================================
# 1. Define the model architecture (same as training)
# ===========================================================
class PSL_Inception(nn.Module):
    def __init__(self, num_classes=40):
        super(PSL_Inception, self).__init__()
        self.model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)

        # Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last block + fc
        for name, param in self.model.named_parameters():
            if "Mixed_7c" in name or "fc" in name:
                param.requires_grad = True

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if self.training:
            outputs, _ = self.model(x)
            return outputs
        else:
            return self.model(x)


def apply_canny_edge_enhancement(image, low_threshold=50, high_threshold=150, edge_weight=0.35):
    rgb = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(rgb, 1.0 - edge_weight, edges_rgb, edge_weight, 0)
    return Image.fromarray(blended)


# ===========================================================
# 2. Image selection & optimized prediction
# ===========================================================
def main():
    root = tk.Tk()
    root.withdraw()
    img_path = filedialog.askopenfilename(title="Select an image for PSL prediction")
    if not img_path:
        print("❌ No image selected.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Running on: {device}")
    torch.backends.cudnn.benchmark = True

    # Load checkpoint first to infer num_classes
    checkpoint = torch.load("C:/Users/qasim/Desktop/static-model.pth", map_location=device, weights_only=False)
    num_classes = checkpoint["model_state_dict"]["model.fc.weight"].shape[0]
    class_names = checkpoint.get("class_names") or [f"class_{i}" for i in range(num_classes)]

    # Build model with correct num_classes
    model = PSL_Inception(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(img_path).convert("RGB")
    image = apply_canny_edge_enhancement(image)
    image_tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)

    if device.type == "cuda":
        for _ in range(2):
            with torch.no_grad():
                _ = model(image_tensor)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)
    end = time.perf_counter()

    print(f"\n🖼️ Image: {img_path}")
    print(f"🔍 Predicted: {class_names[pred.item()]}")
    print(f"📊 Confidence: {conf.item() * 100:.2f}%")
    print(f"⚡ Inference time: {(end - start) * 1000:.2f} ms")


if __name__ == "__main__":
    main()

"""VGG16 model configuration for sign language classification."""

import timm
import torch.nn as nn

MODEL_NAME = "vgg16"
NUM_CLASSES = 36

print(f"Loading {MODEL_NAME} from TIMM...")

model = timm.create_model(
    MODEL_NAME,
    pretrained=True,
    num_classes=NUM_CLASSES,
)

print(f"✅ Model loaded successfully!")
print(f"   Classes: {model.num_classes}")
print(f"   Input size: {model.default_cfg['input_size']}")

print(f"\nFreezing early layers for fine-tuning...")

for param in model.parameters():
    param.requires_grad = False

for param in model.head.parameters():
    param.requires_grad = True

unfrozen_layers = []
for name, param in model.named_parameters():
    if 'head' in name or 'classifier' in name:
        param.requires_grad = True
        unfrozen_layers.append(name)
    elif 'features' in name:
        layer_parts = name.split('.')
        if len(layer_parts) >= 2 and layer_parts[1].isdigit():
            layer_num = int(layer_parts[1])
            if layer_num >= 17:
                param.requires_grad = True
                unfrozen_layers.append(name)

print(f"Unfrozen layers for fine-tuning:")
for layer in unfrozen_layers:
    print(f"   ✓ {layer}")

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTrainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)")
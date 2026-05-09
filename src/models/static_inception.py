import torch.nn as nn
from torchvision.models import Inception_V3_Weights, inception_v3


def build_static_inception(
    num_classes: int,
    freeze_backbone: bool = True,
    unfreeze_mixed7c: bool = True,
) -> nn.Module:
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    if unfreeze_mixed7c:
        for param in model.Mixed_7c.parameters():
            param.requires_grad = True

    in_features_main = model.fc.in_features
    model.fc = nn.Linear(in_features_main, num_classes)

    in_features_aux = model.AuxLogits.fc.in_features
    model.AuxLogits.fc = nn.Linear(in_features_aux, num_classes)

    return model

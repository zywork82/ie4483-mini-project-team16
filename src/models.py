"""Model definitions. Put this file at src/models.py.

Person 2 owns the pretrained models (resnet18, resnet50, vgg16).
Person 3 adds `small_cnn` at the marked spot.

Usage:
    from models import build_model, unfreeze_last_block
    model = build_model("resnet18", num_classes=2)
"""
import torch.nn as nn
from torchvision import models


def _freeze(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def build_model(name: str, num_classes: int = 2, pretrained: bool = True,
                freeze_backbone: bool = True) -> nn.Module:
    """Return a model whose last layer outputs `num_classes` logits.

    freeze_backbone=True : only the new last layer is trained (fast, good start).
    The new last layer is created AFTER freezing, so it is always trainable.
    """
    name = name.lower()

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        if freeze_backbone:
            _freeze(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        if freeze_backbone:
            _freeze(model)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "vgg16":
        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        model = models.vgg16(weights=weights)
        if freeze_backbone:
            _freeze(model)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    
        # ---Person 3: added small cnn ---
    elif name == "small_cnn":
        from small_cnn import SmallCNN
        model = SmallCNN(num_classes=num_classes)  # trained from scratch, nothing frozen
    else:
        raise ValueError(f"Unknown model name: {name}")

    return model

def unfreeze_last_block(model: nn.Module, name: str) -> None:
    """Fine-tuning option: also train the last convolutional block."""
    name = name.lower()
    if name.startswith("resnet"):
        for p in model.layer4.parameters():
            p.requires_grad = True
    elif name == "vgg16":
        for p in model.features[24:].parameters():  # conv block 5
            p.requires_grad = True


if __name__ == "__main__":
    # Quick check: python src/models.py
    from utils import count_parameters
    for n in ["resnet18", "resnet50", "vgg16"]:
        m = build_model(n, num_classes=2, pretrained=False)
        total, trainable = count_parameters(m)
        print(f"{n}: total={total:,} trainable={trainable:,}")

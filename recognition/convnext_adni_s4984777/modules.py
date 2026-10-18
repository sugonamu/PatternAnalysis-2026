
"""
COMP3710 - ADNI Alzheimer's Disease Classification

Models:
    1. BaselineCNN
    2. ConvNeXtClassifier (ConvNeXt-Tiny)

Class mapping:
    0 = Cognitively Normal (NC)
    1 = Alzheimer's Disease (AD)
"""

import torch
import torch.nn as nn

from torchvision.models import (
    convnext_tiny,
    ConvNeXt_Tiny_Weights,
)


# ============================================================
# 1. Baseline CNN
# ============================================================

class BaselineCNN(nn.Module):
    """
    Simple convolutional neural network baseline.

    Input:
        (B, 3, 224, 224)

    Output:
        (B, 2) - raw classification logits
    """

    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            # Convert spatial features to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ============================================================
# 2. ConvNeXt-Tiny Classifier
# ============================================================

class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Tiny adapted for binary ADNI classification.

    Uses the ConvNeXt architecture with an optional
    ImageNet-pretrained initialization.

    Input:
        (B, 3, 224, 224)

    Output:
        (B, 2) - raw classification logits
    """

    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()

        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if pretrained
            else None
        )

        self.model = convnext_tiny(weights=weights)

        # Replace the original ImageNet classification head
        # with a two-class AD/NC classification head.
        in_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
# 3. Model Factory
# ============================================================

def build_model(
    model_name,
    num_classes=2,
    pretrained=True,
):
    """
    Create a model by name.

    Args:
        model_name: "baseline" or "convnext"
        num_classes: Number of output classes
        pretrained: Whether ConvNeXt uses ImageNet weights

    Returns:
        PyTorch model
    """

    model_name = model_name.lower()

    if model_name == "baseline":
        return BaselineCNN(num_classes=num_classes)

    if model_name == "convnext":
        return ConvNeXtClassifier(
            num_classes=num_classes,
            pretrained=pretrained,
        )

    raise ValueError(
        f"Unknown model: {model_name}. "
        "Choose 'baseline' or 'convnext'."
    )


# ============================================================
# 4. Model Testing
# ============================================================

def test_models():
    """Test model output dimensions and parameter counts."""

    # Dummy batch: 2 RGB MRI images
    x = torch.randn(2, 3, 224, 224)

    for name in ("baseline", "convnext"):

        # Avoid downloading pretrained weights during testing.
        model = build_model(
            model_name=name,
            pretrained=False,
        )

        model.eval()

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 2), (
            f"{name}: unexpected output shape "
            f"{tuple(output.shape)}"
        )

        total_params = sum(
            p.numel() for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print(f"\nModel: {name}")
        print(f"Output shape: {tuple(output.shape)}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

    print("\nAll model tests passed!")


if __name__ == "__main__":
    test_models()


"""
Deep Learning Model Architectures for Indian Sign Language Recognition
Includes MobileNetV3-Small, ResNet18, and Custom Multi-Scale CNN.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ISLCustomCNN(nn.Module):
    """
    Lightweight, modular CNN architecture designed specifically for real-time ISL classification.
    """

    def __init__(self, num_classes: int = 26, in_channels: int = 3, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 128x128 -> 64x64
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 64x64 -> 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 32x32 -> 16x16
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 16x16 -> 8x8
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract flat 256-d embedding for temporal sequence modeling."""
        feat = self.features(x)
        flat = torch.flatten(feat, 1)
        return flat


class ISLMobileNetV3(nn.Module):
    """
    MobileNetV3-Small tailored for high-speed, low-latency CPU real-time inference.
    """

    def __init__(self, num_classes: int = 26, pretrained: bool = False, dropout: float = 0.2):
        super().__init__()
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = models.mobilenet_v3_small(weights=weights)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        in_features = base_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ISLResNet18(nn.Module):
    """
    ResNet-18 architecture with residual connections for robust representation learning.
    """

    def __init__(self, num_classes: int = 26, pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_model = models.resnet18(weights=weights)
        
        # Keep conv layers
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        feat = torch.flatten(x, 1)
        out = self.classifier(feat)
        return out

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


def build_model(arch_name: str, num_classes: int = 26, pretrained: bool = False) -> nn.Module:
    """Factory function for static gesture models."""
    arch = arch_name.lower().replace("-", "").replace("_", "")
    if "mobilenet" in arch:
        return ISLMobileNetV3(num_classes=num_classes, pretrained=pretrained)
    elif "resnet" in arch:
        return ISLResNet18(num_classes=num_classes, pretrained=pretrained)
    elif "cnn" in arch:
        return ISLCustomCNN(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch_name}. Supported: mobilenetv3, resnet18, custom_cnn")


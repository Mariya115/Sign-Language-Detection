"""
Image Preprocessing and Augmentation Pipeline for Indian Sign Language Recognition
"""

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import cv2

# ImageNet statistics for transfer learning backbones (ResNet, MobileNetV3, EfficientNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 128):
    """
    Data augmentation pipeline for training ISL classifiers.
    Avoids aggressive horizontal flipping because hand signs in ISL have distinct handedness.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06)),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 128):
    """
    Deterministic preprocessing pipeline for validation, testing, and real-time inference.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def preprocess_cv2_frame(frame: np.ndarray, bbox=None, target_size: int = 128) -> torch.Tensor:
    """
    Preprocess an OpenCV BGR frame (or cropped bbox) directly for model inference.
    Args:
        frame: OpenCV BGR image (H, W, 3)
        bbox: Optional (x1, y1, x2, y2) bounding box
        target_size: Desired model input dimension (default: 128)
    Returns:
        torch.Tensor of shape (1, 3, target_size, target_size) ready for model forward pass
    """
    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            frame = frame[y1:y2, x1:x2]

    # Convert BGR (OpenCV) to RGB (PIL/PyTorch)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transform = get_val_transforms(image_size=target_size)
    tensor = transform(rgb_frame)
    return tensor.unsqueeze(0)  # Add batch dimension: (1, 3, H, W)


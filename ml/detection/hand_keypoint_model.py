"""Dedicated 21-point hand keypoint model.

The detector checkpoint is intentionally not used for keypoints: it is a
YOLO detection model and has no anatomical landmark head.
"""

from typing import Optional

import os
import cv2
import numpy as np
import torch
import torch.nn as nn


class HandKeypointNet(nn.Module):
    """Small coordinate-regression network for 21 normalized hand points."""

    def __init__(self, num_keypoints: int = 21):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_keypoints * 2),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(images)).reshape(-1, 21, 2)


class HandKeypointEstimator:
    """Loads one trained hand-keypoint checkpoint and maps ROI points back."""

    def __init__(self, checkpoint_path: str, device: str = "cpu", input_size: int = 128):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)
        self.input_size = input_size
        self.model: Optional[HandKeypointNet] = None
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model = HandKeypointNet().to(self.device)
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.eval()

    @property
    def is_available(self) -> bool:
        return self.model is not None

    @torch.inference_mode()
    def predict(self, roi: np.ndarray, square_bounds) -> list:
        if self.model is None:
            return []
        x1, y1, x2, y2 = square_bounds
        crop = roi[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        points = self.model(tensor)[0].cpu().numpy()
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        return [
            [int(np.clip(x1 + point[0] * width, x1, x2 - 1)),
             int(np.clip(y1 + point[1] * height, y1, y2 - 1)), 0.0]
            for point in points
        ]

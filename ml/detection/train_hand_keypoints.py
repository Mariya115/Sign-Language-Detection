"""Train the dedicated 21-point hand keypoint model.

Annotation format (JSON list):
[
  {
    "image": "path/to/frame.jpg",
    "bbox": [x1, y1, x2, y2],
    "keypoints": [[x, y], ... 21 points in the required anatomical order]
  }
]
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ml.detection.hand_keypoint_model import HandKeypointNet


class HandKeypointDataset(Dataset):
    def __init__(self, annotation_path: Path, input_size: int = 128):
        self.records = json.loads(annotation_path.read_text(encoding="utf-8"))
        self.input_size = input_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(record["image"])
        if image is None:
            raise RuntimeError(f"Unable to read {record['image']}")
        x1, y1, x2, y2 = [int(value) for value in record["bbox"]]
        crop = image[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        crop = cv2.resize(crop, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        height, width = image.shape[:2]
        points = np.asarray(record["keypoints"], dtype=np.float32)
        points[:, 0] = (points[:, 0] - x1) / max(1, x2 - x1)
        points[:, 1] = (points[:, 1] - y1) / max(1, y2 - y1)
        tensor = torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div(255)
        return tensor, torch.from_numpy(np.clip(points, 0.0, 1.0))


def train(annotation_path: Path, output_path: Path, epochs: int, batch_size: int):
    dataset = HandKeypointDataset(annotation_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    model = HandKeypointNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.SmoothL1Loss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        print(f"epoch {epoch + 1}/{epochs} loss={total_loss / len(loader):.6f}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_keypoints": 21}, output_path)
    print(f"saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("ml/saved_models/hand_keypoints_best.pth"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.annotations, args.output, args.epochs, args.batch_size)

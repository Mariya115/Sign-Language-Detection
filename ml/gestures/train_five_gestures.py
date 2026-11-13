"""Train the five-class landmark sequence classifier."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ml.gestures.labels import GESTURE_NAMES, GESTURE_LABELS
from ml.gestures.landmark_features import sequence_features


class FiveGestureSequenceNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(GESTURE_LABELS)),
        )

    def forward(self, x):
        return self.net(x)


def train(data_dir: Path, output: Path, epochs: int):
    files = sorted(data_dir.glob("*.npy"))
    samples, targets = [], []
    for path in files:
        label = path.stem.rsplit("_", 1)[0]
        if label not in GESTURE_NAMES:
            continue
        samples.append(sequence_features(np.load(path)))
        targets.append(GESTURE_NAMES.index(label))
    if not samples:
        raise FileNotFoundError(f"No five-gesture feature files found in {data_dir}")
    x = torch.from_numpy(np.asarray(samples))
    y = torch.tensor(targets, dtype=torch.long)
    loader = DataLoader(TensorDataset(x, y), batch_size=min(16, len(y)), shuffle=True)
    model = FiveGestureSequenceNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{epochs} loss={float(loss):.5f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(GESTURE_LABELS)}, output)
    print(f"saved {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("ml/gestures/data"))
    parser.add_argument("--output", type=Path, default=Path("ml/saved_models/five_gesture_landmarks.pth"))
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    train(args.data, args.output, args.epochs)

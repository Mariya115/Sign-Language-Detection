"""Train a five-class MLP on normalized MediaPipe landmark features.

Multiple recording sessions are held out by session. A single session uses a
stratified preliminary split because a session-held-out evaluation is not
possible until additional sessions have been collected.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.gestures.labels import GESTURE_LABELS, GESTURE_NAMES


class LandmarkMLP(nn.Module):
    def __init__(self, feature_dim: int = 128, class_count: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, class_count),
        )

    def forward(self, features):
        return self.net(features)


def load_samples(data_dir: Path):
    features, labels, sessions = [], [], []
    seen = set()
    duplicate_count = 0
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            vector = np.asarray(sample["features"], dtype=np.float32)
            label = int(sample["label_index"])
            session = str(sample["session_id"])
        if vector.shape != (128,) or not 0 <= label < len(GESTURE_LABELS):
            continue
        sample_key = (label, vector.tobytes())
        if sample_key in seen:
            duplicate_count += 1
            continue
        seen.add(sample_key)
        features.append(vector)
        labels.append(label)
        sessions.append(session)
    if not features:
        raise FileNotFoundError(f"No valid landmark samples found in {data_dir}")
    return (
        np.asarray(features),
        np.asarray(labels, dtype=np.int64),
        np.asarray(sessions),
        duplicate_count,
    )


def session_split(labels, sessions, test_ratio: float, validation_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    unique_sessions = np.array(sorted(set(sessions)))
    rng.shuffle(unique_sessions)
    test_count = max(1, round(len(unique_sessions) * test_ratio))
    validation_count = max(1, round(len(unique_sessions) * validation_ratio))
    test_sessions = set(unique_sessions[:test_count])
    validation_sessions = set(unique_sessions[test_count:test_count + validation_count])
    train_sessions = set(unique_sessions[test_count + validation_count:])
    if not train_sessions or not validation_sessions or not test_sessions:
        raise ValueError("At least three recording sessions are required.")
    train = np.array([session in train_sessions for session in sessions])
    validation = np.array([session in validation_sessions for session in sessions])
    test = np.array([session in test_sessions for session in sessions])
    expected = set(range(len(GESTURE_LABELS)))
    for name, mask in (("train", train), ("validation", validation), ("test", test)):
        if set(labels[mask]) != expected:
            raise ValueError(f"{name} split must contain all five gesture classes.")
    return train, validation, test


def single_session_split(labels, test_ratio: float, validation_ratio: float, seed: int):
    """Create a stratified random split for preliminary single-session training."""
    rng = np.random.default_rng(seed)
    masks = [
        np.zeros(len(labels), dtype=bool),
        np.zeros(len(labels), dtype=bool),
        np.zeros(len(labels), dtype=bool),
    ]
    split_names = ("train", "validation", "test")
    for class_index in range(len(GESTURE_LABELS)):
        class_indices = np.flatnonzero(labels == class_index)
        rng.shuffle(class_indices)
        test_count = max(1, round(len(class_indices) * test_ratio))
        validation_count = max(1, round(len(class_indices) * validation_ratio))
        train_count = len(class_indices) - test_count - validation_count
        if train_count < 1:
            raise ValueError(
                f"Class {class_index} has only {len(class_indices)} unique samples; "
                "at least three unique samples per class are required."
            )
        boundaries = (train_count, train_count + validation_count)
        masks[0][class_indices[:boundaries[0]]] = True
        masks[1][class_indices[boundaries[0]:boundaries[1]]] = True
        masks[2][class_indices[boundaries[1]:]] = True
    expected = set(range(len(GESTURE_LABELS)))
    for name, mask in zip(split_names, masks):
        if set(labels[mask]) != expected:
            raise ValueError(f"{name} split must contain all five gesture classes.")
    return tuple(masks)


def metrics(actual, predicted):
    matrix = np.zeros((len(GESTURE_LABELS), len(GESTURE_LABELS)), dtype=np.int64)
    for truth, guess in zip(actual, predicted):
        matrix[int(truth), int(guess)] += 1
    precision, recall, f1 = [], [], []
    for index in range(len(GESTURE_LABELS)):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            GESTURE_LABELS[index]: {
                "precision": precision[index],
                "recall": recall[index],
                "f1": f1[index],
            }
            for index in range(len(GESTURE_LABELS))
        },
    }


def train(data_dir: Path, output: Path, report: Path, epochs: int, seed: int):
    torch.manual_seed(seed)
    x, y, sessions, duplicate_count = load_samples(data_dir)
    unique_sessions = np.unique(sessions)
    preliminary = len(unique_sessions) == 1
    if preliminary:
        train_mask, validation_mask, test_mask = single_session_split(
            y, test_ratio=0.15, validation_ratio=0.15, seed=seed
        )
        split_strategy = "stratified_random_single_session"
    else:
        train_mask, validation_mask, test_mask = session_split(
            y, sessions, test_ratio=0.2, validation_ratio=0.2, seed=seed
        )
        split_strategy = "session_held_out"
    model = LandmarkMLP()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    train_x = torch.from_numpy(x[train_mask])
    train_y = torch.from_numpy(y[train_mask])
    validation_x = torch.from_numpy(x[validation_mask])
    validation_y = torch.from_numpy(y[validation_mask])
    best_state = None
    best_validation = -1.0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            validation_prediction = model(validation_x).argmax(dim=1)
        validation_accuracy = float((validation_prediction == validation_y).float().mean())
        if validation_accuracy > best_validation:
            best_validation = validation_accuracy
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        if (epoch + 1) % 10 == 0:
            print(f"epoch={epoch + 1} loss={loss.item():.4f} validation_accuracy={validation_accuracy:.4f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        test_prediction = model(torch.from_numpy(x[test_mask])).argmax(dim=1).numpy()
    result = metrics(y[test_mask], test_prediction)
    payload = {
        "classes": list(GESTURE_LABELS),
        "names": list(GESTURE_NAMES),
        "feature_dim": 128,
        "model_type": "landmark_mlp",
        "state_dict": model.state_dict(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "classes": list(GESTURE_LABELS),
        "evaluation": "preliminary_single_session" if preliminary else "session_held_out",
        "preliminary_evaluation": preliminary,
        "split_strategy": split_strategy,
        "unique_sessions": int(len(unique_sessions)),
        "duplicate_samples_removed": int(duplicate_count),
        "train_samples": int(train_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "test_samples": int(test_mask.sum()),
        "train_class_counts": np.bincount(y[train_mask], minlength=len(GESTURE_LABELS)).tolist(),
        "validation_class_counts": np.bincount(y[validation_mask], minlength=len(GESTURE_LABELS)).tolist(),
        "test_class_counts": np.bincount(y[test_mask], minlength=len(GESTURE_LABELS)).tolist(),
        "test_sessions": sorted(set(sessions[test_mask])),
        "metrics": result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "evaluation": "preliminary_single_session" if preliminary else "session_held_out",
        "split_strategy": split_strategy,
        "duplicate_samples_removed": duplicate_count,
        "metrics": result,
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("ml/gestures/data"))
    parser.add_argument("--output", type=Path, default=Path("ml/saved_models/five_gesture_landmark_mlp.pth"))
    parser.add_argument("--report", type=Path, default=Path("ml/evaluation/five_gesture_mlp_report.json"))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.report, args.epochs, args.seed)

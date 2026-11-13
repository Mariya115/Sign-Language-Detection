"""Compare landmark classifiers using a recording-session-held-out test split."""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.gestures.labels import GESTURE_LABELS, GESTURE_NAMES


def load_samples(data_dir: Path):
    features, targets, groups = [], [], []
    for path in sorted(data_dir.glob("*.npz")):
        with np.load(path, allow_pickle=False) as sample:
            vector = np.asarray(sample["features"], dtype=np.float32)
            label = int(sample["label_index"])
            session = str(sample["session_id"])
        if vector.shape != (128,) or not 0 <= label < len(GESTURE_LABELS):
            continue
        features.append(vector)
        targets.append(label)
        groups.append(session)
    if not features:
        raise FileNotFoundError(f"No valid .npz samples found in {data_dir}")
    return np.asarray(features), np.asarray(targets), np.asarray(groups)


def train(data_dir: Path, output: Path, report_path: Path, test_size: float, seed: int):
    x, y, groups = load_samples(data_dir)
    if len(np.unique(groups)) < 2:
        raise ValueError("At least two recording sessions are required for a held-out test split.")
    splitter = GroupShuffleSplit(n_splits=20, test_size=test_size, random_state=seed)
    train_idx = test_idx = None
    for candidate_train, candidate_test in splitter.split(x, y, groups):
        if set(np.unique(y[candidate_train])) == set(range(len(GESTURE_LABELS))) and set(np.unique(y[candidate_test])) == set(range(len(GESTURE_LABELS))):
            train_idx, test_idx = candidate_train, candidate_test
            break
    if train_idx is None:
        raise ValueError("The session split must contain every gesture class in both train and test data.")

    models = {
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=-1),
        "svm": make_pipeline(StandardScaler(), SVC(C=3.0, probability=True, class_weight="balanced", random_state=seed)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=seed)),
    }
    results = {}
    for name, model in models.items():
        model.fit(x[train_idx], y[train_idx])
        predicted = model.predict(x[test_idx])
        results[name] = {
            "accuracy": float(accuracy_score(y[test_idx], predicted)),
            "precision_macro": float(precision_score(y[test_idx], predicted, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y[test_idx], predicted, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y[test_idx], predicted, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(y[test_idx], predicted, labels=range(len(GESTURE_LABELS))).tolist(),
            "classification_report": classification_report(y[test_idx], predicted, labels=range(len(GESTURE_LABELS)), target_names=GESTURE_LABELS, zero_division=0, output_dict=True),
        }

    best_name = max(results, key=lambda name: (results[name]["f1_macro"], results[name]["accuracy"]))
    best_model = models[best_name]
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "classes": list(GESTURE_LABELS), "names": list(GESTURE_NAMES), "feature_dim": 128, "model_name": best_name}, output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "model_selected": best_name,
        "classes": list(GESTURE_LABELS),
        "train_samples": int(len(train_idx)),
        "test_samples": int(len(test_idx)),
        "train_sessions": sorted(set(groups[train_idx])),
        "test_sessions": sorted(set(groups[test_idx])),
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": best_name, "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("ml/gestures/data"))
    parser.add_argument("--output", type=Path, default=Path("ml/saved_models/five_gesture_landmarks.joblib"))
    parser.add_argument("--report", type=Path, default=Path("ml/evaluation/five_gesture_report.json"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args.data, args.output, args.report, args.test_size, args.seed)

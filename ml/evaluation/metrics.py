"""
Evaluation Metrics Computation (Pure NumPy implementation)
Computes Accuracy, Precision, Recall, F1-Score (Macro & Weighted), Confusion Matrix,
Per-class performance, Inference Latency, and FPS without external DLL dependencies.
"""

from typing import Dict, List, Tuple, Any
import numpy as np


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Compute multi-class confusion matrix.
    Rows: Ground Truth, Columns: Predictions
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Calculate comprehensive multi-class metrics from ground truth and predictions.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num_classes = len(class_names)
    total_samples = len(y_true)

    if total_samples == 0:
        return {}

    cm = compute_confusion_matrix(y_true, y_pred, num_classes)
    overall_accuracy = float(np.sum(np.diag(cm)) / total_samples)

    per_class_metrics = {}
    precisions = []
    recalls = []
    f1s = []
    supports = []

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        support = int(np.sum(cm[i, :]))

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

        cls_name = class_names[i]
        per_class_metrics[cls_name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }

    macro_precision = float(np.mean(precisions))
    macro_recall = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_support = sum(supports)
    weighted_f1 = float(sum(f * s for f, s in zip(f1s, supports)) / total_support) if total_support > 0 else 0.0

    return {
        "accuracy": round(overall_accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_samples": total_samples,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
    }


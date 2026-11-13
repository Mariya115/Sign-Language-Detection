"""
Comprehensive Model Evaluation and Benchmarking Suite
Evaluates trained models on the test dataset, measures latency/FPS,
computes metrics, and generates confusion matrix and training curves.
"""

import os
import sys
import time
import json
from typing import Dict, List, Any
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.datasets.dataset_loader import get_dataloaders
from ml.classification.models import build_model
from ml.evaluation.metrics import evaluate_predictions


def evaluate_checkpoint(
    checkpoint_path: str,
    test_loader,
    classes: List[str],
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Load a trained model checkpoint and evaluate it thoroughly on the test split.
    """
    print(f"[Evaluation] Loading checkpoint: {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    arch_name = checkpoint.get("arch", "mobilenet_v3")

    model = build_model(arch_name, num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    latencies = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            for i in range(images.size(0)):
                single_img = images[i:i+1]
                t0 = time.perf_counter()
                out = model(single_img)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0)  # ms

                _, pred = torch.max(out, 1)
                all_preds.append(int(pred.cpu().item()))
                all_targets.append(int(labels[i].cpu().item()))

    metrics = evaluate_predictions(all_targets, all_preds, class_names=classes)

    mean_latency_ms = float(np.mean(latencies))
    fps = float(1000.0 / mean_latency_ms) if mean_latency_ms > 0 else 0.0

    result = {
        "arch": arch_name,
        "checkpoint": os.path.basename(checkpoint_path),
        "test_accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "mean_latency_ms": round(mean_latency_ms, 2),
        "inference_fps": round(fps, 1),
        "total_test_samples": len(all_targets),
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
    }

    print(
        f"[Evaluation] {arch_name.upper()} Results -> "
        f"Test Accuracy: {result['test_accuracy']*100:.2f}% | "
        f"Macro F1: {result['macro_f1']:.4f} | "
        f"Latency: {result['mean_latency_ms']} ms ({result['inference_fps']} FPS)"
    )

    return result


def plot_confusion_matrix(cm_data: List[List[int]], classes: List[str], save_path: str = "results/confusion_matrix.png"):
    """Render high-resolution confusion matrix heatmap."""
    cm = np.array(cm_data)
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.upper() for c in classes],
        yticklabels=[c.upper() for c in classes],
        cbar=True,
    )
    plt.title("Indian Sign Language Recognition - Test Confusion Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Confusion matrix heatmap saved to {save_path}")


def plot_training_history(history_files: List[str], save_path: str = "results/training_curves.png"):
    """Plot training and validation loss and accuracy curves for compared models."""
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    for hf in history_files:
        if os.path.isfile(hf):
            with open(hf, "r") as f:
                data = json.load(f)
            arch = data["arch"]
            epochs = range(1, len(data["train_loss"]) + 1)
            plt.plot(epochs, data["train_loss"], label=f"{arch} (Train)", linestyle="--")
            plt.plot(epochs, data["val_loss"], label=f"{arch} (Val)", linewidth=2)
    plt.title("Training & Validation Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Accuracy plot
    plt.subplot(1, 2, 2)
    for hf in history_files:
        if os.path.isfile(hf):
            with open(hf, "r") as f:
                data = json.load(f)
            arch = data["arch"]
            epochs = range(1, len(data["train_acc"]) + 1)
            plt.plot(epochs, [a * 100 for a in data["train_acc"]], label=f"{arch} (Train)", linestyle="--")
            plt.plot(epochs, [a * 100 for a in data["val_acc"]], label=f"{arch} (Val)", linewidth=2)
    plt.title("Training & Validation Accuracy (%)", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Evaluation] Training curves saved to {save_path}")


def run_benchmark():
    """Execute benchmark across available saved models."""
    _, _, test_loader, classes = get_dataloaders(batch_size=32, image_size=128)
    models_dir = "ml/saved_models"
    available_checkpoints = [
        os.path.join(models_dir, f) for f in os.listdir(models_dir)
        if f.endswith("_best.pth")
    ]

    all_results = {}
    for cp in available_checkpoints:
        res = evaluate_checkpoint(cp, test_loader, classes)
        all_results[res["arch"]] = res

    # Save comparison JSON
    comp_path = "results/model_comparison.json"
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"[Evaluation] Model comparison saved to {comp_path}")

    # Plot confusion matrix for best model
    if all_results:
        best_arch = max(all_results.keys(), key=lambda k: all_results[k]["test_accuracy"])
        plot_confusion_matrix(all_results[best_arch]["confusion_matrix"], classes)

    # Plot training curves
    history_files = [
        os.path.join("results", f) for f in os.listdir("results")
        if f.endswith("_history.json")
    ]
    if history_files:
        plot_training_history(history_files)

    return all_results


if __name__ == "__main__":
    run_benchmark()


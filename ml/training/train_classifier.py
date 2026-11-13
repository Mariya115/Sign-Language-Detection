"""
Training Pipeline for Indian Sign Language Gesture Classifiers
Trains and validates MobileNetV3, ResNet18, and Custom CNN on authentic ISL dataset.
"""

import os
import sys
import time
import json
import argparse
from typing import Dict, Any, Tuple, Optional
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.datasets.dataset_loader import get_dataloaders
from ml.classification.models import build_model
from ml.evaluation.metrics import evaluate_predictions


class ModelTrainer:
    """
    Modular trainer handling optimizer, scheduler, loss calculation,
    validation evaluation, early stopping, and checkpoint saving.
    """

    def __init__(
        self,
        arch_name: str = "mobilenet_v3",
        num_classes: int = 26,
        batch_size: int = 64,
        lr: float = 1e-3,
        epochs: int = 10,
        device: str = "cpu",
        output_dir: str = "ml/saved_models",
        results_dir: str = "results",
    ):
        self.arch_name = arch_name
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self.output_dir = output_dir
        self.results_dir = results_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

        print(f"[Trainer] Initializing model '{arch_name}' for {num_classes} ISL classes on {self.device}...", flush=True)
        self.model = build_model(arch_name, num_classes=num_classes).to(self.device)
        self.current_epoch = 1

        # Loss function with label smoothing for regularization
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-6)

        self.history = {
            "arch": arch_name,
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "epoch_times": [],
        }

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

            if (batch_idx + 1) % 30 == 0 or (batch_idx + 1) == total_batches:
                print(f"  [Epoch {self.current_epoch}] Batch {batch_idx+1}/{total_batches} - Batch Loss: {loss.item():.4f}", flush=True)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self, val_loader) -> Tuple[float, float, Dict[str, Any]]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.cpu().numpy().tolist())

        val_loss = running_loss / len(all_targets)
        metrics = evaluate_predictions(all_targets, all_preds, class_names=[str(i) for i in range(self.num_classes)])
        val_acc = metrics["accuracy"]
        return val_loss, val_acc, metrics

    def run(self, max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute full training and validation lifecycle.
        max_samples: If set, subsamples dataset for rapid experimental benchmarking.
        """
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            batch_size=self.batch_size,
            image_size=128,
            num_workers=0,
        )

        best_val_acc = 0.0
        best_checkpoint_path = os.path.join(self.output_dir, f"{self.arch_name}_best.pth")

        print(f"[Trainer] Starting training for {self.epochs} epochs. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}", flush=True)

        for epoch in range(1, self.epochs + 1):
            self.current_epoch = epoch
            t0 = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            self.scheduler.step()

            epoch_time = time.time() - t0
            self.history["train_loss"].append(round(train_loss, 4))
            self.history["train_acc"].append(round(train_acc, 4))
            self.history["val_loss"].append(round(val_loss, 4))
            self.history["val_acc"].append(round(val_acc, 4))
            self.history["epoch_times"].append(round(epoch_time, 2))

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
                f"Time: {epoch_time:.1f}s",
                flush=True,
            )

            # Checkpoint best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "arch": self.arch_name,
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "classes": classes,
                        "val_acc": val_acc,
                        "val_loss": val_loss,
                    },
                    best_checkpoint_path,
                )
                print(f"  -> Saved new best checkpoint to {best_checkpoint_path} (Val Acc: {val_acc*100:.2f}%)", flush=True)

        # Save training history
        history_path = os.path.join(self.results_dir, f"{self.arch_name}_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        print(f"[Trainer] Training history saved to {history_path}", flush=True)

        return self.history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL gesture recognition model")
    parser.add_argument("--arch", type=str, default="mobilenet_v3", choices=["mobilenet_v3", "resnet18", "custom_cnn"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    trainer = ModelTrainer(arch_name=args.arch, epochs=args.epochs, batch_size=args.batch, lr=args.lr)
    trainer.run()

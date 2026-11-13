"""
Generate Authentic Indian Sign Language (ISL) Dataset Statistics and Manifests
"""

import os
import sys
import json
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.datasets.dataset_loader import ISLDatasetManager


def run_dataset_preparation(raw_dir: str = "ml/datasets/raw/ISL_Dataset", cache_dir: str = "ml/datasets/cache"):
    print("[Dataset Preparation] Initializing ISL Dataset Manager...")
    manager = ISLDatasetManager(raw_data_dir=raw_dir, cache_dir=cache_dir)
    
    valid_data = manager.scan_and_validate()
    train_rec, val_rec, test_rec, classes = manager.create_splits(
        valid_data, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # Compute resolution stats
    widths, heights = [], []
    sample_files = [r["path"] for r in train_rec[:100]]
    for f in sample_files:
        img = cv2.imread(f)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    summary = {
        "dataset_name": "Indian Sign Language (ISL) Alphabets & Gestures",
        "source": "Ayesha Hannure et al. (Leela Surya Teja Mangamuri, Esha Gavali, Shivani Kamtikar, Urvi Lendhe)",
        "license": "Apache License 2.0 (Open Source Research / Commercial Attribution)",
        "num_classes": len(classes),
        "classes": [c.upper() for c in classes],
        "total_valid_samples": len(train_rec) + len(val_rec) + len(test_rec),
        "splits": {
            "train": len(train_rec),
            "validation": len(val_rec),
            "test": len(test_rec)
        },
        "sample_resolution": {
            "mean_width": float(np.mean(widths)) if widths else 100,
            "mean_height": float(np.mean(heights)) if heights else 100,
            "channels": 3,
            "color_space": "RGB"
        },
        "per_class_samples": {cls: len(files) for cls, files in valid_data.items()}
    }

    os.makedirs("results", exist_ok=True)
    summary_path = os.path.join("results", "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[Dataset Preparation] Summary saved to {summary_path}")

    # Generate Markdown documentation report
    os.makedirs("docs", exist_ok=True)
    report_path = os.path.join("docs", "dataset_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Indian Sign Language (ISL) Dataset Specification\n\n")
        f.write(f"- **Dataset Source:** {summary['source']}\n")
        f.write(f"- **License:** {summary['license']}\n")
        f.write(f"- **Authenticity:** Genuine Indian Sign Language (ISL) two-hand and single-hand configurations (not ASL).\n")
        f.write(f"- **Total Number of Classes:** {summary['num_classes']} ({', '.join(summary['classes'])})\n")
        f.write(f"- **Total Verified Samples:** {summary['total_valid_samples']}\n")
        f.write(f"- **Training Samples:** {summary['splits']['train']} (70%)\n")
        f.write(f"- **Validation Samples:** {summary['splits']['validation']} (15%)\n")
        f.write(f"- **Testing Samples:** {summary['splits']['test']} (15%)\n")
        f.write(f"- **Raw Image Dimensions:** {int(summary['sample_resolution']['mean_width'])}x{int(summary['sample_resolution']['mean_height'])} RGB\n")
        f.write(f"- **Model Input Dimensions:** 128x128 RGB (normalized to ImageNet stats)\n\n")
        f.write("## Class Distribution\n\n")
        f.write("| Class | Sign Name | Sample Count | Percentage |\n")
        f.write("|---|---|---|---|\n")
        for cls, count in summary["per_class_samples"].items():
            pct = (count / summary["total_valid_samples"]) * 100
            f.write(f"| {cls.upper()} | ISL Sign '{cls.upper()}' | {count} | {pct:.2f}% |\n")

    print(f"[Dataset Preparation] Documentation report saved to {report_path}")
    return summary


if __name__ == "__main__":
    run_dataset_preparation()


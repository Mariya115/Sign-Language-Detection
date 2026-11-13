"""
Custom YOLO Hand Detection Training and Fine-Tuning Workflow
Provides end-to-end dataset preparation and training using Ultralytics YOLO.
"""

import os
import sys
import yaml
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def generate_hand_dataset_yaml(output_yaml_path: str = "ml/detection/hand_data.yaml") -> str:
    """
    Generate dataset YAML configuration conforming to Ultralytics YOLO requirements.
    """
    config = {
        "path": os.path.abspath("ml/datasets/detection_data"),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "hand"
        }
    }
    os.makedirs(os.path.dirname(output_yaml_path), exist_ok=True)
    with open(output_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"[YOLO Training] Generated dataset config: {output_yaml_path}")
    return output_yaml_path


def train_custom_hand_detector(
    dataset_yaml: str = "ml/detection/hand_data.yaml",
    base_model: str = "yolov8n.pt",
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    output_dir: str = "ml/saved_models",
):
    """
    Fine-tune YOLO model on custom hand bounding box annotations.
    """
    print(f"[YOLO Training] Initializing fine-tuning with base model: {base_model}")
    model = YOLO(base_model)
    
    # Train model
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=output_dir,
        name="custom_hand_yolo",
        exist_ok=True,
        verbose=True,
    )

    best_weights = os.path.join(output_dir, "custom_hand_yolo", "weights", "best.pt")
    print(f"[YOLO Training] Training finished. Best weights saved to: {best_weights}")
    return best_weights


if __name__ == "__main__":
    yaml_path = generate_hand_dataset_yaml()
    print("Hand detection configuration ready. To train custom weights on new annotations, run with data.")


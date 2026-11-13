"""
Indian Sign Language Dataset Loader, Splitting, and Integrity Validation
"""

import os
import json
import hashlib
import random
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ml.preprocessing.transforms import get_train_transforms, get_val_transforms


def compute_file_md5(filepath: str) -> str:
    """Compute MD5 checksum of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class ISLDatasetManager:
    """
    Manages raw ISL datasets: scanning, integrity validation, duplicate removal,
    train/val/test splitting, and manifest persistence.
    """

    def __init__(self, raw_data_dir: str, cache_dir: str = "ml/datasets/cache", seed: int = 42):
        self.raw_data_dir = raw_data_dir
        self.cache_dir = cache_dir
        self.seed = seed
        os.makedirs(self.cache_dir, exist_ok=True)

    def scan_and_validate(self) -> Dict[str, List[str]]:
        """
        Scan classes, verify readability, and filter corrupted files.
        Returns mapping: {class_label: [valid_file_paths]}
        """
        valid_data: Dict[str, List[str]] = {}
        seen_hashes = set()
        duplicates_count = 0
        corrupt_count = 0

        classes = sorted([
            d for d in os.listdir(self.raw_data_dir)
            if os.path.isdir(os.path.join(self.raw_data_dir, d))
        ])

        for cls_name in classes:
            cls_folder = os.path.join(self.raw_data_dir, cls_name)
            valid_files = []
            for fname in sorted(os.listdir(cls_folder)):
                fpath = os.path.join(cls_folder, fname)
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue

                # Check readability & non-empty
                if os.path.getsize(fpath) == 0:
                    corrupt_count += 1
                    continue

                img = cv2.imread(fpath)
                if img is None or img.size == 0:
                    corrupt_count += 1
                    continue

                # Check duplicates via md5
                fhash = compute_file_md5(fpath)
                if fhash in seen_hashes:
                    duplicates_count += 1
                    continue
                seen_hashes.add(fhash)
                valid_files.append(fpath)

            valid_data[cls_name] = valid_files

        print(f"[DatasetManager] Scan complete: {len(valid_data)} classes, {duplicates_count} duplicates skipped, {corrupt_count} corrupt skipped.")
        return valid_data

    def create_splits(
        self,
        valid_data: Dict[str, List[str]],
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[str]]:
        """
        Create stratified train/val/test splits without data leakage.
        """
        random.seed(self.seed)
        classes = sorted(list(valid_data.keys()))
        label_to_idx = {cls: idx for idx, cls in enumerate(classes)}

        train_records = []
        val_records = []
        test_records = []

        for cls_name, files in valid_data.items():
            shuffled_files = list(files)
            random.shuffle(shuffled_files)
            n = len(shuffled_files)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_subset = shuffled_files[:n_train]
            val_subset = shuffled_files[n_train:n_train + n_val]
            test_subset = shuffled_files[n_train + n_val:]

            cls_idx = label_to_idx[cls_name]
            for f in train_subset:
                train_records.append({"path": f, "label": cls_name, "class_idx": cls_idx})
            for f in val_subset:
                val_records.append({"path": f, "label": cls_name, "class_idx": cls_idx})
            for f in test_subset:
                test_records.append({"path": f, "label": cls_name, "class_idx": cls_idx})

        # Save manifests
        self._save_json(os.path.join(self.cache_dir, "classes.json"), classes)
        self._save_json(os.path.join(self.cache_dir, "train_manifest.json"), train_records)
        self._save_json(os.path.join(self.cache_dir, "val_manifest.json"), val_records)
        self._save_json(os.path.join(self.cache_dir, "test_manifest.json"), test_records)

        print(f"[DatasetManager] Manifests saved: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
        return train_records, val_records, test_records, classes

    def _save_json(self, path: str, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class ISLImageDataset(Dataset):
    """
    PyTorch Dataset for Indian Sign Language images loaded from a manifest.
    """

    def __init__(self, manifest: List[Dict], transform=None):
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        record = self.manifest[idx]
        image_path = record["path"]
        class_idx = record["class_idx"]

        bgr_image = cv2.imread(image_path)
        if bgr_image is None:
            # Fallback black image if corrupt
            bgr_image = np.zeros((128, 128, 3), dtype=np.uint8)

        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            tensor = self.transform(rgb_image)
        else:
            default_t = get_val_transforms(128)
            tensor = default_t(rgb_image)

        return tensor, class_idx


def get_dataloaders(
    cache_dir: str = "ml/datasets/cache",
    batch_size: int = 32,
    image_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Construct train, validation, and test PyTorch DataLoaders from cached manifests.
    """
    with open(os.path.join(cache_dir, "classes.json"), "r", encoding="utf-8") as f:
        classes = json.load(f)
    with open(os.path.join(cache_dir, "train_manifest.json"), "r", encoding="utf-8") as f:
        train_manifest = json.load(f)
    with open(os.path.join(cache_dir, "val_manifest.json"), "r", encoding="utf-8") as f:
        val_manifest = json.load(f)
    with open(os.path.join(cache_dir, "test_manifest.json"), "r", encoding="utf-8") as f:
        test_manifest = json.load(f)

    train_ds = ISLImageDataset(train_manifest, transform=get_train_transforms(image_size))
    val_ds = ISLImageDataset(val_manifest, transform=get_val_transforms(image_size))
    test_ds = ISLImageDataset(test_manifest, transform=get_val_transforms(image_size))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, classes


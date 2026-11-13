"""
Phase 2 Tests: ISL Dataset Loading, Manifests, Transforms, and Class Balance
"""

import os
import sys
import json
import unittest
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.datasets.dataset_loader import get_dataloaders
from ml.preprocessing.transforms import preprocess_cv2_frame
import numpy as np


class TestPhase2(unittest.TestCase):
    def test_manifests_exist(self):
        """Verify cached manifests and classes exist and contain all 26 classes."""
        cache_dir = os.path.join(BASE_DIR, "ml", "datasets", "cache")
        classes_path = os.path.join(cache_dir, "classes.json")
        train_path = os.path.join(cache_dir, "train_manifest.json")
        val_path = os.path.join(cache_dir, "val_manifest.json")
        test_path = os.path.join(cache_dir, "test_manifest.json")

        self.assertTrue(os.path.isfile(classes_path))
        self.assertTrue(os.path.isfile(train_path))
        self.assertTrue(os.path.isfile(val_path))
        self.assertTrue(os.path.isfile(test_path))

        with open(classes_path, "r", encoding="utf-8") as f:
            classes = json.load(f)
        self.assertEqual(len(classes), 26)

    def test_dataloaders_batch(self):
        """Verify DataLoader batch loading and tensor dimensions."""
        train_loader, val_loader, test_loader, classes = get_dataloaders(
            cache_dir=os.path.join(BASE_DIR, "ml", "datasets", "cache"),
            batch_size=8,
            image_size=128,
            num_workers=0,
        )
        self.assertGreater(len(train_loader), 0)
        self.assertGreater(len(val_loader), 0)
        self.assertGreater(len(test_loader), 0)

        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape, (8, 3, 128, 128))
        self.assertEqual(labels.shape, (8,))
        self.assertEqual(images.dtype, torch.float32)

    def test_cv2_frame_preprocessing(self):
        """Verify preprocess_cv2_frame formats arbitrary frame into normalized tensor."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_bbox = (100, 100, 300, 300)
        tensor = preprocess_cv2_frame(dummy_frame, bbox=dummy_bbox, target_size=128)
        self.assertEqual(tensor.shape, (1, 3, 128, 128))
        self.assertEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()


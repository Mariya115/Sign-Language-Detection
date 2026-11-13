"""
Phase 3 Tests: YOLO Hand Detection, Cropping, and Bounding-Box Handling
"""

import os
import sys
import unittest
import numpy as np
import cv2

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.detection.hand_detector import YOLOHandDetector


class TestPhase3(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        weights_path = os.path.join(BASE_DIR, "ml", "saved_models", "hand_yolov8n.pt")
        cls.detector = YOLOHandDetector(weights_path=weights_path, conf_threshold=0.25, target_size=128)

    def test_detector_initialization(self):
        """Verify YOLO hand detector is loaded and recognizes hand class."""
        self.assertIsNotNone(self.detector.model)
        self.assertEqual(self.detector.class_names[0], "hand")

    def test_no_hand_detection(self):
        """Verify graceful handling when no hand is present in frame."""
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(black_frame)
        self.assertFalse(result["detected"])
        self.assertEqual(result["num_hands"], 0)
        self.assertEqual(len(result["boxes"]), 0)
        self.assertEqual(len(result["crops"]), 0)
        self.assertIsNone(result["combined_crop"])

    def test_hand_detection_on_isl_sample(self):
        """Verify detection, bounding box extraction, and square cropping on authentic ISL sample."""
        sample_path = os.path.join(BASE_DIR, "ml", "datasets", "raw", "ISL_Dataset", "a", "A1.jpg")
        frame = cv2.imread(sample_path)
        self.assertIsNotNone(frame)

        result = self.detector.detect(frame)
        self.assertTrue(result["detected"])
        self.assertGreaterEqual(result["num_hands"], 1)

        # Check crop shapes
        for crop in result["crops"]:
            self.assertEqual(crop.shape, (128, 128, 3))
        self.assertEqual(result["combined_crop"].shape, (128, 128, 3))

    def test_draw_detections(self):
        """Verify visualization overlay draws without changing frame dimensions."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_result = {
            "detected": True,
            "num_hands": 1,
            "boxes": [[50, 50, 200, 200, 0.95]]
        }
        annotated = self.detector.draw_detections(frame, dummy_result)
        self.assertEqual(annotated.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()


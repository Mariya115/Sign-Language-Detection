"""
Phase 1 Smoke Tests: Project Setup, Dependencies, Structure, and Health Check
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


class TestPhase1(unittest.TestCase):
    def test_required_imports(self):
        """Verify that all core computer vision and deep learning packages import cleanly."""
        import torch
        import torchvision
        import ultralytics
        import cv2
        import fastapi
        import uvicorn
        import gtts
        import pyttsx3
        import PIL
        import numpy

        self.assertIsNotNone(torch.__version__)
        self.assertIsNotNone(torchvision.__version__)
        self.assertIsNotNone(ultralytics.__version__)
        self.assertIsNotNone(cv2.__version__)
        self.assertIsNotNone(fastapi.__version__)

    def test_directory_structure(self):
        """Verify all project directories exist according to the modular architecture specification."""
        required_dirs = [
            "backend/api",
            "backend/services",
            "backend/camera",
            "backend/recognition",
            "ml/datasets",
            "ml/preprocessing",
            "ml/detection",
            "ml/classification",
            "ml/temporal",
            "ml/training",
            "ml/evaluation",
            "ml/saved_models",
            "notebooks",
            "results",
            "docs",
            "tests",
            "frontend/src/components",
            "frontend/src/pages",
            "frontend/src/services",
        ]
        for d in required_dirs:
            full_path = os.path.join(BASE_DIR, d)
            self.assertTrue(os.path.isdir(full_path), f"Required directory missing: {d}")

    def test_backend_health_endpoint(self):
        """Verify FastAPI backend health check endpoint directly."""
        from backend.main import health_check

        direct_data = health_check()
        self.assertEqual(direct_data["status"], "healthy")
        self.assertIn("torch", direct_data["versions"])
        self.assertIn("opencv", direct_data["versions"])
        self.assertEqual(direct_data["device"], "cpu")

    def test_frontend_build_exists(self):
        """Verify frontend production build was generated."""
        dist_index = os.path.join(BASE_DIR, "frontend", "dist", "index.html")
        self.assertTrue(os.path.isfile(dist_index), "Frontend dist/index.html was not generated")


if __name__ == "__main__":
    unittest.main()

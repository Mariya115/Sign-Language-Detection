"""
Interactive Indian Sign Language Data Collection and Recording Studio
Allows capturing and labeling new ISL gestures (alphabets, numbers, words) via webcam.
"""

import os
import time
import argparse
import cv2
import numpy as np


class ISLDataCollector:
    """
    Modular webcam recorder for capturing, cropping, and saving ISL gesture datasets.
    """

    def __init__(self, output_dir: str = "ml/datasets/raw/ISL_Dataset", target_size: int = 128):
        self.output_dir = output_dir
        self.target_size = target_size
        os.makedirs(self.output_dir, exist_ok=True)

    def record_sign(
        self,
        sign_label: str,
        num_samples: int = 100,
        camera_id: int = 0,
        delay_seconds: int = 3,
        auto_capture_interval: float = 0.15,
    ) -> int:
        """
        Record images for a given sign label using the webcam.
        """
        class_dir = os.path.join(self.output_dir, sign_label.lower())
        os.makedirs(class_dir, exist_ok=True)
        existing_count = len([f for f in os.listdir(class_dir) if f.endswith(".jpg")])

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"[DataCollector] Error: Could not open camera {camera_id}")
            return 0

        print(f"[DataCollector] Preparing to record '{sign_label}'. Starting in {delay_seconds} seconds...")
        start_time = time.time()
        while time.time() - start_time < delay_seconds:
            ret, frame = cap.read()
            if not ret:
                break
            remaining = int(delay_seconds - (time.time() - start_time)) + 1
            cv2.putText(
                frame,
                f"Starting in {remaining}s... Prepare gesture '{sign_label}'",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow("ISL Gesture Data Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                return 0

        collected = 0
        last_capture = 0.0

        while collected < num_samples:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            # Center region of interest (ROI)
            roi_size = min(h, w) // 2
            x1, y1 = (w - roi_size) // 2, (h - roi_size) // 2
            x2, y2 = x1 + roi_size, y1 + roi_size

            # Save frame crop at intervals
            now = time.time()
            if now - last_capture >= auto_capture_interval:
                crop = frame[y1:y2, x1:x2]
                resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
                sample_idx = existing_count + collected + 1
                save_path = os.path.join(class_dir, f"{sign_label}_{sample_idx:04d}.jpg")
                cv2.imwrite(save_path, resized)
                collected += 1
                last_capture = now

            # Visualization overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Recording: {sign_label} ({collected}/{num_samples})",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("ISL Gesture Data Collector", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"[DataCollector] Successfully recorded {collected} samples for '{sign_label}' into {class_dir}")
        return collected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Record ISL gestures via webcam")
    parser.add_argument("--sign", type=str, required=True, help="Name of the sign (e.g. namaste, thank_you, a, b)")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to capture")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    args = parser.parse_args()

    collector = ISLDataCollector()
    collector.record_sign(args.sign, num_samples=args.samples, camera_id=args.cam)


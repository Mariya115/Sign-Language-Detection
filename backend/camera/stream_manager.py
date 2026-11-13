"""
Camera Stream Manager
Threaded asynchronous OpenCV camera acquisition, MJPEG streaming,
frame-skipping inference pipeline, and synthetic fallback pattern.
"""

import time
import threading
from typing import Optional, Dict, Any, Generator
import cv2
import numpy as np

from backend.recognition.inference_engine import RealTimeISLEngine
from backend.services.text_to_speech import SentenceBuilder


class CameraStreamManager:
    """
    Manages continuous video capture from webcam or synthetic test source.
    Runs asynchronous frame reading thread and integrates RealTimeISLEngine.
    """

    def __init__(self, inference_engine: Optional[RealTimeISLEngine] = None, frame_skip: int = 1):
        self.engine = inference_engine or RealTimeISLEngine()
        self.sentence_builder = SentenceBuilder()
        self.frame_skip = max(1, frame_skip)

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.camera_id = 0
        self.lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.frame_event = threading.Event()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None
        self.latest_input_frame: Optional[np.ndarray] = None
        self.latest_input_counter = 0

        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.latest_result: Dict[str, Any] = {
            "prediction": "Camera Stopped",
            "confidence": 0.0,
            "fps": 0.0,
            "hand_detected": False,
            "num_hands": 0,
            "boxes": [],
            "landmarks": [],
            "landmarks_detected": False,
            "landmark_count": 0,
            "gesture_recognized": False,
            "predicted_class": None,
            "kannada_label": None,
            "status": "camera_stopped",
            "latency_ms": 0.0,
        }

        self.use_synthetic_source = False
        self.frame_counter = 0
        self.latest_input_frame = None
        self.latest_input_counter = 0

    def start(self, camera_id: int = 0) -> bool:
        """Start camera capture thread."""
        with self.lock:
            if self.is_running:
                return True

            self.camera_id = camera_id
            self.cap = cv2.VideoCapture(camera_id)
            self.stop_event.clear()
            self.frame_counter = 0
            self.latest_result = {
                "prediction": "Starting Camera",
                "confidence": 0.0,
                "fps": 0.0,
                "hand_detected": False,
                "num_hands": 0,
                "boxes": [],
                "landmarks": [],
                "landmarks_detected": False,
                "landmark_count": 0,
                "gesture_recognized": False,
                "predicted_class": None,
                "kannada_label": None,
                "status": "starting",
                "latency_ms": 0.0,
            }

            if not self.cap.isOpened():
                print(f"[CameraManager] Webcam {camera_id} not accessible. Falling back to synthetic test stream.")
                self.use_synthetic_source = True
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.use_synthetic_source = False

            self.is_running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.thread.start()
            self.inference_thread.start()
            print(f"[CameraManager] Camera loop started (id={camera_id}, synthetic={self.use_synthetic_source})")
            return True

    def stop(self):
        """Stop camera capture thread."""
        with self.lock:
            self.is_running = False
            self.stop_event.set()
            self.frame_event.set()
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.latest_result["prediction"] = "Camera Stopped"
            print("[CameraManager] Camera stopped.")

    def _capture_loop(self):
        """Read frames continuously without waiting for inference."""
        synthetic_step = 0
        read_failures = 0

        while self.is_running:
            if self.use_synthetic_source:
                frame = self._generate_synthetic_frame(synthetic_step)
                synthetic_step += 1
                time.sleep(0.033)  # ~30 FPS
            else:
                ret, frame = self.cap.read()
                if not ret:
                    read_failures += 1
                    if read_failures >= 30:
                        print("[CameraManager] Camera reads are failing. Switching to synthetic source.")
                        self.use_synthetic_source = True
                    time.sleep(0.01)
                    continue
                read_failures = 0

            self.frame_counter += 1
            with self.frame_lock:
                self.latest_input_frame = frame
                self.latest_input_counter = self.frame_counter
            with self.lock:
                self.latest_raw_frame = frame
            self.frame_event.set()

    def _inference_loop(self):
        """Process only the newest frame so slow inference never freezes capture."""
        processed_counter = 0
        while self.is_running:
            self.frame_event.wait(timeout=0.1)
            self.frame_event.clear()
            if self.stop_event.is_set():
                return

            with self.frame_lock:
                frame = self.latest_input_frame
                counter = self.latest_input_counter
            if frame is None or counter == processed_counter:
                continue
            if counter % self.frame_skip != 0:
                continue
            processed_counter = counter

            try:
                result = self.engine.process_frame(frame)
            except Exception as exc:
                print(f"[CameraManager] Inference error: {exc}")
                with self.lock:
                    self.latest_annotated_frame = frame.copy()
                    self.latest_result = {
                        **self.latest_result,
                        "prediction": "Inference Error",
                        "error": str(exc),
                        "hand_detected": False,
                    }
                continue

            if result["status"] == "recognized" and result["prediction"]:
                self.sentence_builder.add_prediction(
                    result["prediction"],
                    result["confidence"],
                    is_word=False,
                )

            with self.lock:
                self.latest_annotated_frame = result["annotated_frame"]
                self.latest_result = {
                    "prediction": result["prediction"],
                    "confidence": result["confidence"],
                    "raw_confidence": result["raw_confidence"],
                    "probabilities": result["probabilities"],
                    "fps": result["fps"],
                    "latency_ms": result["latency_ms"],
                    "hand_detected": result["hand_detected"],
                    "num_hands": result["num_hands"],
                    "boxes": result["boxes"],
                    "hand_detection_confidence": result.get("hand_detection_confidence", 0.0),
                    "landmarks": result.get("landmarks", []),
                    "landmarks_detected": result.get("landmarks_detected", False),
                    "landmark_count": result.get("landmark_count", 0),
                    "gesture_recognized": result.get("gesture_recognized", False),
                    "predicted_class": result.get("predicted_class"),
                    "kannada_label": result.get("kannada_label"),
                    "english_label": result.get("english_label"),
                    "status": result.get("status", "unknown"),
                    "kannada_prediction": result.get("kannada_label"),
                    "gesture": result.get("gesture"),
                    "sentence": self.sentence_builder.get_sentence(),
                }

    def _generate_synthetic_frame(self, step: int) -> np.ndarray:
        """
        Generates clean animated video test pattern when physical webcam is absent.
        """
        frame = np.full((480, 640, 3), 18, dtype=np.uint8)

        # Draw grid
        for x in range(0, 640, 40):
            cv2.line(frame, (x, 0), (x, 480), (30, 41, 59), 1)
        for y in range(0, 480, 40):
            cv2.line(frame, (0, y), (640, y), (30, 41, 59), 1)

        # Animated simulated hand marker
        cx = int(320 + 80 * np.sin(step * 0.05))
        cy = int(240 + 40 * np.cos(step * 0.05))
        cv2.circle(frame, (cx, cy), 50, (46, 204, 113), -1)
        cv2.putText(frame, "ISL TEST", (cx - 38, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        cv2.putText(frame, "SYNTHETIC TEST CAMERA", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 116, 139), 1)
        return frame

    def get_latest_result(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.latest_result)

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yield MJPEG multipart stream chunks for HTTP video streaming.
        """
        while True:
            with self.lock:
                frame = self.latest_annotated_frame if self.latest_annotated_frame is not None else self.latest_raw_frame

            if frame is None:
                # Standby frame
                frame = np.full((480, 640, 3), 15, dtype=np.uint8)
                cv2.putText(frame, "CAMERA OFF", (230, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 116, 139), 2)

            ret, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
            time.sleep(0.033)

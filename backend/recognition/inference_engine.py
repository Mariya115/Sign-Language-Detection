"""
Real-Time ISL Inference Engine
Integrates YOLO Hand Detection, Image Preprocessing, Deep Learning Classifier,
Confidence Thresholding, Temporal Smoothing, and Dynamic Gesture Buffer.
"""

import time
import os
import json
from collections import deque, Counter
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import joblib

from ml.detection.hand_detector import YOLOHandDetector
from ml.classification.models import build_model
from ml.preprocessing.transforms import preprocess_cv2_frame
from ml.temporal.temporal_models import TemporalSequenceBuffer
from ml.gestures.labels import GESTURE_LABELS, english_label_for_class
from ml.gestures.landmark_features import frame_features, hands_to_feature_input, sequence_features
from ml.gestures.train_five_gestures import FiveGestureSequenceNet
from ml.gestures.train_landmark_mlp import LandmarkMLP


class RealTimeISLEngine:
    """
    End-to-End Real-Time Indian Sign Language Recognition Engine.
    Fully integrates camera frame processing, hand localization, gesture classification,
    flicker-free temporal smoothing, and confidence gating.
    """

    def __init__(
        self,
        classifier_checkpoint: str = "ml/saved_models/custom_cnn_best.pth",
        gesture_checkpoint: str = "ml/saved_models/five_gesture_landmarks.pth",
        landmark_mlp_checkpoint: str = "ml/saved_models/five_gesture_landmark_mlp.pth",
        landmark_classifier_checkpoint: str = "ml/saved_models/five_gesture_landmarks.joblib",
        hand_detector_weights: str = "ml/saved_models/hand_yolov8n.pt",
        classes_manifest: str = "ml/datasets/cache/classes.json",
        conf_threshold: float = 0.55,
        hand_conf_threshold: float = 0.30,
        smoothing_window: int = 5,
        target_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.smoothing_window = smoothing_window
        self.target_size = target_size
        self.gesture_checkpoint = gesture_checkpoint
        self.gesture_model = None
        self.gesture_model_available = False
        self.landmark_classifier = None
        self.landmark_classifier_name = None
        self.landmark_mlp = None

        # Load classes
        if os.path.isfile(classes_manifest):
            with open(classes_manifest, "r", encoding="utf-8") as f:
                self.classes = json.load(f)
        else:
            self.classes = [chr(ord('a') + i) for i in range(26)]
        self.num_classes = len(self.classes)

        # 1. Initialize YOLO hand detector and MediaPipe landmark provider
        self.detector = YOLOHandDetector(
            weights_path=hand_detector_weights,
            conf_threshold=hand_conf_threshold,
            target_size=target_size,
            device=device,
            keypoint_weights_path="ml/saved_models/hand_keypoints_best.pth",
        )

        # 2. Initialize Classifier
        self.classifier_checkpoint = classifier_checkpoint
        self.model = None
        self.load_classifier(classifier_checkpoint)
        if os.path.isfile(landmark_mlp_checkpoint):
            checkpoint = torch.load(landmark_mlp_checkpoint, map_location=self.device)
            if checkpoint.get("feature_dim") != 128 or checkpoint.get("classes") != list(GESTURE_LABELS):
                raise ValueError(f"Incompatible landmark MLP checkpoint: {landmark_mlp_checkpoint}")
            self.landmark_mlp = LandmarkMLP().to(self.device)
            self.landmark_mlp.load_state_dict(checkpoint["state_dict"])
            self.landmark_mlp.eval()
            self.gesture_model_available = True
            print("[InferenceEngine] Loaded five-class landmark MLP successfully.")
        if os.path.isfile(landmark_classifier_checkpoint):
            package = joblib.load(landmark_classifier_checkpoint)
            if package.get("feature_dim") != 128 or package.get("classes") != list(GESTURE_LABELS):
                raise ValueError(f"Incompatible landmark classifier: {landmark_classifier_checkpoint}")
            self.landmark_classifier = package["model"]
            self.landmark_classifier_name = package.get("model_name")
            self.gesture_model_available = True
        if os.path.isfile(gesture_checkpoint) and self.detector.mp_hands is not None:
            checkpoint = torch.load(gesture_checkpoint, map_location=self.device)
            if checkpoint.get("classes") == list(GESTURE_LABELS):
                self.gesture_model = FiveGestureSequenceNet().to(self.device)
                self.gesture_model.load_state_dict(checkpoint["state_dict"])
                self.gesture_model.eval()
                self.gesture_model_available = True
            else:
                print("[InferenceEngine] Ignoring gesture checkpoint with incompatible class mapping.")

        # 3. Smoothing Deque for flicker reduction
        self.prediction_buffer = deque(maxlen=smoothing_window)
        self.confidence_buffer = deque(maxlen=smoothing_window)

        # 4. Temporal Sequence Buffer for dynamic signs (16 frames)
        self.sequence_buffer = TemporalSequenceBuffer(sequence_length=32, feature_dim=128)

        # 5. FPS & Performance tracking
        self.fps = 0.0
        self.frame_times = deque(maxlen=30)
        self.total_frames_processed = 0
        self._last_debug_state = None

    def load_classifier(self, checkpoint_path: str):
        """Load or reload model weights."""
        if not os.path.isfile(checkpoint_path):
            print(f"[InferenceEngine] Warning: Checkpoint {checkpoint_path} not found yet. Using untrained model.")
            self.model = build_model("custom_cnn", num_classes=self.num_classes).to(self.device)
            self.model.eval()
            return

        print(f"[InferenceEngine] Loading classifier checkpoint: {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        arch = checkpoint.get("arch", "custom_cnn")
        if "classes" in checkpoint:
            self.classes = checkpoint["classes"]
            self.num_classes = len(self.classes)

        self.model = build_model(arch, num_classes=self.num_classes).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.classifier_checkpoint = checkpoint_path
        print(f"[InferenceEngine] Loaded {arch.upper()} classifier successfully!")

    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Process a single video frame end-to-end.
        """
        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        # Step 1: Detect hands using YOLO
        det_result = self.detector.detect(frame)
        # Keep the semantic hand state aligned with the visual overlays. A
        # valid MediaPipe landmark set is sufficient evidence even if YOLO's
        # boolean result is unavailable for a frame.
        hand_detected = bool(det_result.get("detected") or det_result.get("landmarks"))
        num_hands = det_result["num_hands"]

        prediction_label = "No Hand Detected" if not hand_detected else "Sign Not Recognized"
        display_label = prediction_label
        raw_confidence = 0.0
        smoothed_confidence = 0.0
        probabilities = {}
        hand_crop = None
        status = "no_hand"
        predicted_class = None

        if not hand_detected:
            self.prediction_buffer.clear()
            self.confidence_buffer.clear()
            self.sequence_buffer.clear()
            status = "no_hand"
        elif not det_result.get("landmarks_detected"):
            self.sequence_buffer.clear()
            prediction_label = "Landmarks Not Detected"
            display_label = prediction_label
            status = "landmarks_unavailable"
        elif not self.gesture_model_available:
            prediction_label = "Gesture Model Not Trained"
            display_label = prediction_label
            status = "gesture_model_not_trained"
        else:
            points = det_result["landmarks"]
            landmark_vector = frame_features(
                hands_to_feature_input(points, det_result.get("landmark_handedness"))
            )
            print_state = (
                f"hand={hand_detected} landmarks={len(points)} "
                f"features={landmark_vector.shape}"
            )
            if print_state != self._last_debug_state:
                print(f"[InferenceEngine] {print_state}")
                self._last_debug_state = print_state
            if self.landmark_mlp is not None:
                with torch.inference_mode():
                    logits = self.landmark_mlp(torch.from_numpy(landmark_vector).unsqueeze(0).to(self.device))
                    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                predicted_class = int(np.argmax(probabilities))
                raw_confidence = float(probabilities[predicted_class])
                self._apply_prediction_smoothing(predicted_class, raw_confidence)
                predicted_class, raw_confidence = self._smoothed_prediction(predicted_class, raw_confidence)
                if raw_confidence >= self.conf_threshold:
                    prediction_label = GESTURE_LABELS[predicted_class]
                    display_label = prediction_label
                    smoothed_confidence = raw_confidence
                    status = "recognized"
                else:
                    prediction_label = "Gesture Not Recognized"
                    display_label = prediction_label
                    status = "not_recognized"
            elif self.landmark_classifier is not None:
                probabilities = self.landmark_classifier.predict_proba(landmark_vector.reshape(1, -1))[0]
                predicted_class = int(np.argmax(probabilities))
                raw_confidence = float(probabilities[predicted_class])
                self._apply_prediction_smoothing(predicted_class, raw_confidence)
                predicted_class, raw_confidence = self._smoothed_prediction(predicted_class, raw_confidence)
                if raw_confidence >= self.conf_threshold:
                    prediction_label = GESTURE_LABELS[predicted_class]
                    display_label = prediction_label
                    smoothed_confidence = raw_confidence
                    status = "recognized"
                else:
                    prediction_label = "Gesture Not Recognized"
                    display_label = prediction_label
                    status = "not_recognized"
            else:
                self.sequence_buffer.add_frame_features(torch.from_numpy(landmark_vector))
                sequence = self.sequence_buffer.get_sequence_tensor()
                if sequence is not None and self.gesture_model is not None:
                    with torch.inference_mode():
                        gesture_input = torch.from_numpy(
                            sequence_features(sequence[0].numpy())
                        ).unsqueeze(0).to(self.device)
                        logits = self.gesture_model(gesture_input)
                        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    predicted_class = int(np.argmax(probabilities))
                    raw_confidence = float(probabilities[predicted_class])
                    if raw_confidence >= self.conf_threshold:
                        prediction_label = GESTURE_LABELS[predicted_class]
                        display_label = prediction_label
                        smoothed_confidence = raw_confidence
                        status = "recognized"
                    else:
                        prediction_label = "Gesture Not Recognized"
                        display_label = prediction_label
                        status = "not_recognized"
                else:
                    status = "buffering"

        debug_prediction = (status, predicted_class, round(raw_confidence, 4), prediction_label)
        if debug_prediction != getattr(self, "_last_debug_prediction", None):
            print(
                f"[InferenceEngine] prediction status={status} class={predicted_class} "
                f"confidence={raw_confidence:.4f} output={prediction_label}"
            )
            self._last_debug_prediction = debug_prediction

        # Step 7: Render visual annotations
        annotated_frame = self.detector.draw_detections(frame, det_result)

        # Draw HUD status header
        hud_color = (46, 204, 113) if hand_detected else (52, 152, 219)
        if display_label == "Gesture Not Recognized":
            hud_color = (231, 76, 60)

        # FPS calculation
        t_elapsed = time.perf_counter() - t_start
        self.frame_times.append(t_elapsed)
        avg_time = np.mean(self.frame_times) if self.frame_times else 0.033
        self.fps = round(1.0 / avg_time, 1) if avg_time > 0 else 30.0
        self.total_frames_processed += 1

        # Draw status bar at top
        cv2.rectangle(annotated_frame, (0, 0), (w, 50), (15, 23, 42), -1)
        status_text = f"Sign: {display_label} ({smoothed_confidence*100:.1f}%)" if hand_detected else "Status: Place Hand in View"
        cv2.putText(annotated_frame, status_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, hud_color, 2, cv2.LINE_AA)
        
        fps_text = f"FPS: {self.fps} | Hands: {num_hands}"
        (fw, _), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.putText(annotated_frame, fps_text, (w - fw - 20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1, cv2.LINE_AA)

        if isinstance(probabilities, dict):
            probability_map = {
                str(label): round(float(probability), 4)
                for label, probability in probabilities.items()
            }
        else:
            probability_map = {
                GESTURE_LABELS[index]: round(float(probability), 4)
                for index, probability in enumerate(np.asarray(probabilities).reshape(-1))
                if index < len(GESTURE_LABELS)
            }

        return {
            "annotated_frame": annotated_frame,
            "hand_detected": hand_detected,
            "num_hands": num_hands,
            "boxes": det_result["boxes"],
            "hand_detection_confidence": max((box[4] for box in det_result["boxes"]), default=0.0),
            "landmarks": det_result.get("landmarks", []),
            "landmarks_detected": det_result.get("landmarks_detected", False),
            "landmark_count": det_result.get("landmark_count", 0),
            "landmark_model_available": det_result.get("landmark_model_available", False),
            "prediction": display_label,
            "predicted_class": predicted_class,
            "kannada_label": prediction_label if status == "recognized" else None,
            "english_label": english_label_for_class(predicted_class) if status == "recognized" and predicted_class is not None else None,
            "gesture_recognized": status == "recognized",
            "status": status,
            "raw_prediction": prediction_label,
            "confidence": round(smoothed_confidence, 4),
            "raw_confidence": round(raw_confidence, 4),
            "probabilities": probability_map,
            "fps": self.fps,
            "latency_ms": round(t_elapsed * 1000.0, 2),
            "sequence_ready": self.sequence_buffer.is_ready(),
        }

    def _apply_prediction_smoothing(self, predicted_class: int, confidence: float) -> None:
        self.prediction_buffer.append(predicted_class)
        self.confidence_buffer.append(confidence)

    def _smoothed_prediction(self, predicted_class: int, confidence: float) -> Tuple[int, float]:
        counts = Counter(self.prediction_buffer)
        stable_class, stable_count = counts.most_common(1)[0]
        if stable_count < max(2, (len(self.prediction_buffer) + 1) // 2):
            return predicted_class, confidence
        stable_confidence = np.mean([
            value for prediction, value in zip(self.prediction_buffer, self.confidence_buffer)
            if prediction == stable_class
        ])
        return stable_class, float(stable_confidence)

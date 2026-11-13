"""
YOLO-based Hand Detection and Region Extraction Module
Uses Ultralytics YOLO for detection and MediaPipe Hands for real 21-point
landmark extraction on the detected ROI.
"""

import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import cv2
from ultralytics import YOLO

try:
    import mediapipe as mp
except ImportError:
    mp = None


class YOLOHandDetector:
    """
    Real-time Hand Detector utilizing Ultralytics YOLOv8.
    Supports single-hand, dual-hand (crucial for ISL two-handed signs), and no-hand detection.
    """

    def __init__(
        self,
        weights_path: str = "ml/saved_models/hand_yolov8n.pt",
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45,
        target_size: int = 128,
        device: str = "cpu",
        keypoint_weights_path: str = "ml/saved_models/hand_keypoints_best.pth",
        debug_landmarks: bool = False,
    ):
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.target_size = target_size
        self.device = device
        self.debug_landmarks = debug_landmarks
        self.mp_hands = None
        if mp is not None:
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=0,
                min_detection_confidence=0.45,
                min_tracking_confidence=0.45,
            )
        
        print(f"[YOLOHandDetector] Loading hand detection model from {weights_path}...")
        self.model = YOLO(weights_path)
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Detect hands in a frame.
        Args:
            frame: OpenCV BGR frame (H, W, 3)
        Returns:
            Dictionary containing:
                - 'detected': bool (True if >=1 hand found)
                - 'num_hands': int
                - 'boxes': List of [x1, y1, x2, y2, conf]
                - 'crops': List of square-cropped, resized hand images (target_size, target_size, 3)
                - 'combined_crop': Cropped region enclosing all detected hands (ideal for dual-hand ISL signs)
                - 'latency_ms': float
        """
        start_t = time.perf_counter()
        h, w = frame.shape[:2]

        results = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=320,
            max_det=2,
            device=self.device,
            verbose=False,
        )

        boxes = []
        crops = []
        combined_crop = None

        if len(results) > 0 and len(results[0].boxes) > 0:
            raw_boxes = results[0].boxes
            all_x1, all_y1, all_x2, all_y2 = [], [], [], []

            for b in raw_boxes:
                xyxy = b.xyxy[0].cpu().numpy().astype(int)
                conf = float(b.conf[0].cpu().numpy())
                x1, y1, x2, y2 = xyxy

                # Clamp to frame boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 > x1 and y2 > y1:
                    boxes.append([x1, y1, x2, y2, conf])
                    all_x1.append(x1)
                    all_y1.append(y1)
                    all_x2.append(x2)
                    all_y2.append(y2)

                    # Extract square-padded crop for individual hand
                    crop = self._extract_square_crop(frame, x1, y1, x2, y2)
                    crops.append(crop)

            # Combined bounding box for multi-hand signs (e.g. two-handed ISL alphabets like 'A', 'B', 'P')
            if all_x1:
                cx1 = min(all_x1)
                cy1 = min(all_y1)
                cx2 = max(all_x2)
                cy2 = max(all_y2)
                combined_crop = self._extract_square_crop(frame, cx1, cy1, cx2, cy2)

        landmarks = []
        landmark_handedness = []
        landmark_boxes = []
        if self.mp_hands is not None:
            for x1, y1, x2, y2, _ in boxes:
                roi, bounds = self._extract_padded_roi(frame, x1, y1, x2, y2)
                points, side = self._extract_roi_landmarks(roi, bounds)
                if len(points) == 21:
                    landmarks.append(points)
                    landmark_handedness.append(side)
                    landmark_boxes.append(list(bounds))

            # A hand detector and landmark detector have different failure
            # modes. Full-frame fallback is useful when the YOLO box is tight
            # or MediaPipe cannot resolve the cropped ROI.
            if boxes and not landmarks:
                full_points = self._extract_full_frame_landmarks(frame)
                for points, side in full_points:
                    landmarks.append(points)
                    landmark_handedness.append(side)
                    landmark_boxes.append([
                        min(point[0] for point in points),
                        min(point[1] for point in points),
                        max(point[0] for point in points),
                        max(point[1] for point in points),
                    ])
            elif not boxes:
                full_points = self._extract_full_frame_landmarks(frame)
                for points, side in full_points:
                    landmarks.append(points)
                    landmark_handedness.append(side)
                    landmark_boxes.append([
                        min(point[0] for point in points),
                        min(point[1] for point in points),
                        max(point[0] for point in points),
                        max(point[1] for point in points),
                    ])

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "detected": bool(boxes or landmarks),
            "num_hands": max(len(boxes), len(landmarks)),
            "boxes": boxes,
            "crops": crops,
            "combined_crop": combined_crop if combined_crop is not None else (crops[0] if crops else None),
            "landmarks": landmarks,
            "landmarks_detected": bool(landmarks),
            "landmark_count": 21 * len(landmarks),
            "landmark_model_available": self.mp_hands is not None,
            "landmark_boxes": landmark_boxes,
            "landmark_handedness": landmark_handedness,
            "latency_ms": latency_ms,
        }

    def _extract_padded_roi(self, frame, x1, y1, x2, y2, pad_ratio=0.25):
        height, width = frame.shape[:2]
        side = max(x2 - x1, y2 - y1)
        pad = int(side * pad_ratio)
        side += 2 * pad
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sx1 = max(0, cx - side // 2)
        sy1 = max(0, cy - side // 2)
        sx2 = min(width, sx1 + side)
        sy2 = min(height, sy1 + side)
        return frame[sy1:sy2, sx1:sx2], (sx1, sy1, sx2, sy2)

    def _extract_roi_landmarks(self, roi: np.ndarray, bounds) -> Tuple[List[List[float]], str]:
        if self.mp_hands is None or roi.size == 0:
            return [], ""
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)
        if not result.multi_hand_landmarks:
            return [], ""
        sx1, sy1, sx2, sy2 = bounds
        roi_height, roi_width = roi.shape[:2]
        hand = result.multi_hand_landmarks[0]
        side = ""
        if result.multi_handedness:
            side = result.multi_handedness[0].classification[0].label.lower()
        return [
            [
                int(np.clip(sx1 + point.x * roi_width, 0, sx2 - 1)),
                int(np.clip(sy1 + point.y * roi_height, 0, sy2 - 1)),
                float(point.z),
            ]
            for point in hand.landmark
        ], side

    def _extract_full_frame_landmarks(self, frame: np.ndarray) -> List[Tuple[List[List[float]], str]]:
        if self.mp_hands is None:
            return []
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)
        height, width = frame.shape[:2]
        return [
            ([
                [
                    int(np.clip(point.x * width, 0, width - 1)),
                    int(np.clip(point.y * height, 0, height - 1)),
                    float(point.z),
                ]
                for point in hand.landmark
            ], result.multi_handedness[index].classification[0].label.lower() if result.multi_handedness and index < len(result.multi_handedness) else "")
            for index, hand in enumerate(result.multi_hand_landmarks or [])
        ]

    def _extract_square_crop(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_ratio: float = 0.15) -> np.ndarray:
        """
        Extract a square crop with padding around bounding box to avoid aspect ratio distortion.
        """
        h, w = frame.shape[:2]
        bw = x2 - x1
        bh = y2 - y1

        # Calculate square dimension with padding
        max_dim = max(bw, bh)
        pad = int(max_dim * pad_ratio)
        side = max_dim + 2 * pad

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        new_x1 = max(0, center_x - side // 2)
        new_y1 = max(0, center_y - side // 2)
        new_x2 = min(w, new_x1 + side)
        new_y2 = min(h, new_y1 + side)

        crop = frame[new_y1:new_y2, new_x1:new_x2]
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            return np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        resized = cv2.resize(crop, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        return resized

    def draw_detections(self, frame: np.ndarray, detection_result: Dict[str, Any]) -> np.ndarray:
        """
        Draw clean bounding boxes and status labels on frame.
        """
        annotated = frame.copy()
        boxes = detection_result.get("boxes", [])

        for b in boxes:
            x1, y1, x2, y2, conf = b
            # Emerald bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (46, 204, 113), 2)
            label = f"Hand: {conf*100:.1f}%"
            # Background pill for label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), (46, 204, 113), -1)
            cv2.putText(annotated, label, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        connections = (
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17),
        )
        for points in detection_result.get("landmarks", []):
            for index, (x, y, _) in enumerate(points):
                color = (0, 0, 255) if index == 0 else (255, 190, 0) if index in (4, 8, 12, 16, 20) else (255, 220, 80)
                cv2.circle(annotated, (x, y), 4 if index in (0, 4, 8, 12, 16, 20) else 3, color, -1)
            for start, end in connections:
                cv2.line(annotated, (points[start][0], points[start][1]), (points[end][0], points[end][1]), (255, 190, 0), 2)
            if self.debug_landmarks:
                for index, (x, y, _) in enumerate(points):
                    cv2.putText(annotated, str(index), (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        return annotated

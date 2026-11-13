"""Collect session-labelled, normalized 21-point landmark samples."""

import argparse
import sys
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.detection.hand_detector import YOLOHandDetector
from ml.gestures.landmark_features import frame_features, hands_to_feature_input
from ml.gestures.labels import GESTURE_NAMES, GESTURE_LABELS

LABELS = GESTURE_NAMES


def collect(label: str, samples: int, output: Path, camera_index: int, session_id: str):
    if label not in LABELS:
        raise ValueError(f"label must be one of: {', '.join(LABELS)}")
    detector = YOLOHandDetector(keypoint_weights_path="ml/saved_models/hand_keypoints_best.pth")
    if detector.mp_hands is None:
        raise RuntimeError("Install mediapipe before collecting gesture features")
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")
    output.mkdir(parents=True, exist_ok=True)
    saved = 0
    recent = deque(maxlen=30)
    label_index = LABELS.index(label)
    try:
        while saved < samples:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.putText(frame, f"{label} {saved}/{samples} SPACE=capture Q=quit",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("Five Gesture Collector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key != 32:
                continue
            result = detector.detect(frame)
            hands = result.get("landmarks", [])
            if not hands or len(hands[0]) != 21:
                print("Skipped: no complete 21-point hand landmarks.")
                continue
            features = frame_features(
                hands_to_feature_input(hands, result.get("landmark_handedness"))
            )
            if recent and min(float(np.linalg.norm(features - previous)) for previous in recent) < 0.01:
                print("Skipped duplicate/near-duplicate sample.")
                continue
            recent.append(features)
            np.savez_compressed(
                output / f"{label}_{session_id}_{int(time.time() * 1000)}.npz",
                features=features,
                label_index=label_index,
                label_name=label,
                kannada_label=GESTURE_LABELS[label_index],
                session_id=session_id,
            )
            saved += 1
            print(f"Captured {saved}/{samples} for {label}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=LABELS)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--session-id", default=f"session-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--output", type=Path, default=Path("ml/gestures/data"))
    args = parser.parse_args()
    collect(args.label, args.samples, args.output, args.camera, args.session_id)

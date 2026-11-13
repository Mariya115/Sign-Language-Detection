"""Fast, fixed-size features for two-hand landmark sequences."""

from typing import Dict, Iterable, List, Optional

import numpy as np

FEATURE_DIM = 128
LANDMARK_COUNT = 21


def normalize_hand(points: np.ndarray) -> np.ndarray:
    """Translate a 21x3 hand to the wrist and normalize its scale."""
    if points.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected {LANDMARK_COUNT} landmarks with x/y/z coordinates, got {points.shape}")
    normalized = points.astype(np.float32).copy()
    wrist = normalized[0].copy()
    normalized -= wrist
    scale = float(np.linalg.norm(normalized[9, :2]))
    if scale < 1e-4:
        scale = 1.0
    normalized[:, :2] /= scale
    normalized[:, 2] /= scale
    return normalized


def hands_to_feature_input(landmarks: Iterable[np.ndarray], handedness: Optional[Iterable[str]] = None) -> Dict[str, Optional[np.ndarray]]:
    """Place real MediaPipe hands into stable left/right feature slots."""
    slots: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
    labels = list(handedness or [])
    for index, points in enumerate(landmarks):
        points_array = np.asarray(points, dtype=np.float32)
        if points_array.shape != (LANDMARK_COUNT, 3):
            continue
        side = labels[index].lower() if index < len(labels) else ""
        if side not in slots:
            side = "left" if slots["left"] is None else "right"
        if slots[side] is None:
            slots[side] = points_array
    return slots


def frame_features(hands: Dict[str, Optional[np.ndarray]]) -> np.ndarray:
    """Return left/right normalized landmarks and visibility masks."""
    values: List[float] = []
    for name in ("left", "right"):
        points = hands.get(name)
        if points is None:
            values.extend([0.0] * 63)
        else:
            values.extend(normalize_hand(points).reshape(-1).tolist())
    values.extend([float(hands.get("left") is not None), float(hands.get("right") is not None)])
    return np.asarray(values, dtype=np.float32)


def sequence_features(frames: Iterable[np.ndarray]) -> np.ndarray:
    """Stack frame vectors and append first-order motion features."""
    sequence = np.asarray(list(frames), dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIM:
        raise ValueError(f"Expected (frames, {FEATURE_DIM}) landmark features")
    velocity = np.diff(sequence, axis=0, prepend=sequence[:1])
    return np.concatenate([sequence, velocity], axis=1)

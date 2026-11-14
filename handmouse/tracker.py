"""Hand landmark detection with MediaPipe."""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def ensure_model(path: Path | None = None) -> Path:
    """Return the hand landmark model, downloading it (about 8 MB) on first use."""
    if path is not None:
        return path
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "handmouse"
    path = cache / "hand_landmarker.task"
    if not path.exists():
        print(f"Downloading hand tracking model to {path} ...", file=sys.stderr)
        cache.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(MODEL_URL, partial)
        partial.replace(path)
    return path


class HandTracker:
    def __init__(self, model_path: Path):
        self._landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
            )
        )
        self._last_ms = -1

    def detect(self, frame_bgr, t: float) -> list[tuple[float, float]] | None:
        """Return the hand's 21 landmarks as fractions of the frame, or None."""
        # MediaPipe rejects timestamps that do not strictly increase.
        self._last_ms = max(int(t * 1000), self._last_ms + 1)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        result = self._landmarker.detect_for_video(image, self._last_ms)
        if not result.hand_landmarks:
            return None
        return [(p.x, p.y) for p in result.hand_landmarks[0]]

    def close(self) -> None:
        self._landmarker.close()

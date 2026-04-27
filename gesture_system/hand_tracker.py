from __future__ import annotations

import time
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


class HandTracker:
    def __init__(self, detection_confidence: float, tracking_confidence: float) -> None:
        # "backend" tells us which MediaPipe API is available in this environment.
        self.backend = "unknown"
        # Fallback hand skeleton edges for tasks mode custom drawing.
        self._connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]

        # Preferred legacy API when available.
        if hasattr(mp, "solutions"):
            self.backend = "solutions"
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = self.mp_hands.Hands(
                max_num_hands=1,
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            return

        # Newer MediaPipe builds expose only Tasks API.
        if hasattr(mp, "tasks"):
            self.backend = "tasks"
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            model_path = self._ensure_task_model()
            options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=model_path),
                num_hands=1,
                min_hand_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
                running_mode=vision.RunningMode.VIDEO,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
            # VIDEO mode requires strictly increasing timestamps.
            self._last_timestamp_ms = 0
            return

        raise RuntimeError("Unsupported mediapipe build: no solutions/tasks API found.")

    @staticmethod
    def _ensure_task_model() -> str:
        # Tasks API needs a .task model file. Download once if missing.
        model_path = Path("models/hand_landmarker.task")
        if model_path.exists():
            return str(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        url = (
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task"
        )
        try:
            urlretrieve(url, model_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download models/hand_landmarker.task automatically. "
                "Please download it manually and place it in the models folder."
            ) from exc
        return str(model_path)

    def process(self, frame):
        # MediaPipe expects RGB frames.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.backend == "solutions":
            return self.hands.process(rgb)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # Keep timestamp monotonic for detect_for_video.
        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms
        return self.landmarker.detect_for_video(mp_image, timestamp_ms)

    def extract_landmarks(self, frame, results):
        # Convert MediaPipe outputs into a unified internal hand format.
        if self.backend == "solutions":
            if not results.multi_hand_landmarks:
                return []

            h, w, _ = frame.shape
            detected_hands = []
            handedness = results.multi_handedness or []

            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
                side = "Unknown"
                if idx < len(handedness):
                    side = handedness[idx].classification[0].label
                detected_hands.append({"side": side, "landmarks": points, "raw": hand_landmarks})
            return detected_hands

        if not results or not getattr(results, "hand_landmarks", None):
            return []

        h, w, _ = frame.shape
        detected_hands = []
        for idx, hand_landmarks in enumerate(results.hand_landmarks):
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            side = "Unknown"
            if getattr(results, "handedness", None) and idx < len(results.handedness):
                side = results.handedness[idx][0].category_name
            detected_hands.append({"side": side, "landmarks": points, "raw": points})
        return detected_hands

    def draw(self, frame, hand):
        # Use native drawing when available; fallback to manual drawing for tasks mode.
        if self.backend == "solutions":
            self.mp_draw.draw_landmarks(frame, hand["raw"], self.mp_hands.HAND_CONNECTIONS)
            return

        for a, b in self._connections:
            cv2.line(frame, hand["landmarks"][a], hand["landmarks"][b], (0, 255, 0), 2)
        for pt in hand["landmarks"]:
            cv2.circle(frame, pt, 3, (0, 0, 255), -1)

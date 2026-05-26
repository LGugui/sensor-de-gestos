import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

_COLORS = {
    "Right": (0, 255, 0),
    "Left":  (0, 180, 255),
}


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo de rastreamento de maos (~7MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo baixado.")


class HandDetector:
    def __init__(self, config):
        _ensure_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=config["max_hands"],
            min_hand_detection_confidence=config["detection_confidence"],
            min_tracking_confidence=config["tracking_confidence"],
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self._start = time.time()
        self._flip_handedness = config.get("flip_handedness", True)

    def find_hands(self, frame, draw=True):
        """Returns (frame, hands) where hands = list of (landmarks, label).
        label is 'Left' or 'Right' already adjusted for camera flip."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._start) * 1000)
        results = self.detector.detect_for_video(mp_image, ts_ms)

        hands = []
        if results.hand_landmarks:
            for lm, hd in zip(results.hand_landmarks, results.handedness):
                raw_label = hd[0].category_name  # "Left" or "Right"
                # Camera is flipped so MediaPipe labels are mirrored — invert
                if self._flip_handedness:
                    label = "Left" if raw_label == "Right" else "Right"
                else:
                    label = raw_label
                hands.append((lm, label))
                if draw:
                    self._draw(frame, lm, label)

        return frame, hands

    def _draw(self, frame, landmarks, label="Right"):
        h, w = frame.shape[:2]
        color = _COLORS.get(label, (0, 255, 0))
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)
        for x, y in pts:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
        # label tag near wrist
        cv2.putText(frame, label, (pts[0][0] + 8, pts[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def close(self):
        self.detector.close()

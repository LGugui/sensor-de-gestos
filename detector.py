import math
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

# Landmarks that are fingertips — drawn larger and highlighted
_FINGERTIPS = {4, 8, 12, 16, 20}
# Pinch pair: thumb tip + index tip
_PINCH_PAIR = (4, 8)
# Visual threshold: below this distance (normalized) = fingertips touching
_TOUCH_VISUAL_DIST = 0.04


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

        # Check pinch contact for visual highlight
        t4, t8 = landmarks[_PINCH_PAIR[0]], landmarks[_PINCH_PAIR[1]]
        pinch_dist = math.sqrt((t4.x - t8.x)**2 + (t4.y - t8.y)**2)
        pinch_contact = pinch_dist < _TOUCH_VISUAL_DIST

        # Connections
        for a, b in _CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, 2)

        # Pinch contact line between thumb tip and index tip
        if pinch_contact:
            cv2.line(frame, pts[4], pts[8], (0, 0, 255), 3)

        # Draw each landmark
        for i, (x, y) in enumerate(pts):
            is_tip = i in _FINGERTIPS
            is_pinch_tip = i in _PINCH_PAIR

            if is_tip:
                radius = 12
                if is_pinch_tip and pinch_contact:
                    cv2.circle(frame, (x, y), radius + 6, (0, 0, 255), 2)
                    dot_color = (0, 0, 255)
                else:
                    # Outer precision ring
                    cv2.circle(frame, (x, y), radius + 4, (0, 180, 180), 1)
                    dot_color = (0, 255, 255)

                # Crosshair lines through fingertip for precision targeting
                arm = radius + 8
                cv2.line(frame, (x - arm, y), (x + arm, y), (0, 0, 0), 3)
                cv2.line(frame, (x, y - arm), (x, y + arm), (0, 0, 0), 3)
                cv2.line(frame, (x - arm, y), (x + arm, y), dot_color, 1)
                cv2.line(frame, (x, y - arm), (x, y + arm), dot_color, 1)
            else:
                radius = 4
                dot_color = (255, 255, 255)

            cv2.circle(frame, (x, y), radius, dot_color, -1)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1)

            # Landmark number
            nx = x + radius + 2
            ny = y - 2
            cv2.putText(frame, str(i), (nx, ny),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (220, 220, 0), 1)

        # Contact midpoint indicator
        if pinch_contact:
            mx = (pts[4][0] + pts[8][0]) // 2
            my = (pts[4][1] + pts[8][1]) // 2
            cv2.circle(frame, (mx, my), 6, (0, 0, 255), -1)
            cv2.putText(frame, "PINCH", (mx + 8, my + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Label near wrist
        cv2.putText(frame, label, (pts[0][0] + 8, pts[0][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def close(self):
        self.detector.close()

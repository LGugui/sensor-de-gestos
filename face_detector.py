import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# Iris landmark indices in the 478-point face model
_IRIS_LEFT  = 468
_IRIS_RIGHT = 472


def _ensure_face_model():
    if not os.path.exists(FACE_MODEL_PATH):
        print("Baixando modelo facial (~6MB)...")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        print("Modelo facial baixado.")


class FaceDetector:
    def __init__(self, config):
        _ensure_face_model()
        base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=config.get("face_detection_confidence", 0.5),
            min_face_presence_confidence=0.5,
            min_tracking_confidence=config.get("face_tracking_confidence", 0.5),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self._start = time.time()

    def find_face(self, frame):
        """Returns (face_landmarks, blendshapes, transform_matrix) or (None,None,None)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.time() - self._start) * 1000)
        results = self.detector.detect_for_video(mp_image, ts_ms)

        if not results.face_landmarks:
            return None, None, None

        lm = results.face_landmarks[0]
        bs = results.face_blendshapes[0] if results.face_blendshapes else []

        matrix = None
        if results.facial_transformation_matrixes:
            raw = results.facial_transformation_matrixes[0]
            matrix = np.array(raw).reshape(4, 4) if not isinstance(raw, np.ndarray) else raw

        return lm, bs, matrix

    def close(self):
        self.detector.close()

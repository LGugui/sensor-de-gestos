"""
Gesture classifier backed by scikit-learn MLP + joblib.
TensorFlow not required — works on Python 3.14.
"""
import math
import os

MODEL_PATH  = os.path.join(os.path.dirname(__file__), "gesture_model.pkl")
CLASSES_PATH = os.path.join(os.path.dirname(__file__), "gesture_classes.txt")

GESTURE_CLASSES = [
    "neutral",      # 0
    "pinch_left",   # 1  polegar + indicador
    "pinch_right",  # 2  polegar + médio
    "fist",         # 3  punho fechado
    "open_palm",    # 4  palma aberta
    "precision",    # 5  semi-pinch
    "scroll",       # 6  indicador + médio eretos
    "ring_only",    # 7  só anelar
    "pinky_only",   # 8  só mindinho
]


def normalize_lm(lm):
    """21 landmarks → 63 floats. Translate to wrist origin, scale by palm size."""
    bx, by = lm[0].x, lm[0].y
    scale = math.hypot(lm[9].x - bx, lm[9].y - by)
    if scale < 1e-6:
        scale = 1.0
    coords = []
    for pt in lm:
        coords.extend([
            (pt.x - bx) / scale,
            (pt.y - by) / scale,
            pt.z / scale,
        ])
    return coords  # 63 values


class GestureClassifier:
    def __init__(self, model_path=MODEL_PATH):
        import joblib
        import numpy as np
        self._model = joblib.load(model_path)
        self._np = np
        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                self._classes = [ln.strip() for ln in f if ln.strip()]
        else:
            self._classes = GESTURE_CLASSES
        print(f"[GestureClassifier] modelo carregado: {model_path}")
        print(f"[GestureClassifier] classes: {self._classes}")

    def classify(self, lm):
        """Returns dict[class_name → confidence float]."""
        features = normalize_lm(lm)
        inp = self._np.array([features])
        probs = self._model.predict_proba(inp)[0]
        return {cls: float(p) for cls, p in zip(self._classes, probs)}

    def top(self, lm):
        """Returns (class_name, confidence) of highest-probability class."""
        probs = self.classify(lm)
        best = max(probs, key=probs.get)
        return best, probs[best]


# Alias kept so gestures.py import doesn't need changing
TFGestureClassifier = GestureClassifier

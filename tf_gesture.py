"""
TFLite gesture classifier. Loaded optionally by GestureDetector.
If gesture_model.tflite is absent, gestures.py falls back to rule-based.
"""
import math
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gesture_model.tflite")
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


class TFGestureClassifier:
    def __init__(self, model_path=MODEL_PATH):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        import numpy as np
        import tensorflow as tf

        self._np = np
        self._interp = tf.lite.Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        inp = self._interp.get_input_details()
        out = self._interp.get_output_details()
        self._in_idx = inp[0]["index"]
        self._out_idx = out[0]["index"]

        if os.path.exists(CLASSES_PATH):
            with open(CLASSES_PATH) as f:
                self._classes = [l.strip() for l in f if l.strip()]
        else:
            self._classes = GESTURE_CLASSES

        print(f"[TFGesture] modelo carregado: {model_path}")
        print(f"[TFGesture] classes: {self._classes}")

    def classify(self, lm):
        """Returns dict[class_name → confidence]."""
        features = normalize_lm(lm)
        inp = self._np.array([features], dtype=self._np.float32)
        self._interp.set_tensor(self._in_idx, inp)
        self._interp.invoke()
        probs = self._interp.get_tensor(self._out_idx)[0]
        return {cls: float(p) for cls, p in zip(self._classes, probs)}

    def top(self, lm):
        """Returns (class_name, confidence) of the top prediction."""
        probs = self.classify(lm)
        best = max(probs, key=probs.get)
        return best, probs[best]

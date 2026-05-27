import cv2
import numpy as np


class Camera:
    def __init__(self, index=0):
        self._mode = "screen" if index == "screen" else "webcam"
        if self._mode == "screen":
            try:
                import mss
                self._mss = mss.mss()
                # monitors[0] = full virtual desktop (all monitors combined)
                self._monitor = self._mss.monitors[0]
            except ImportError:
                raise RuntimeError(
                    "mss nao instalado. Execute: pip install mss"
                )
        else:
            self._mss = None
            self.cap = cv2.VideoCapture(index)
            if not self.cap.isOpened():
                raise RuntimeError(f"Camera {index} nao encontrada.")

    def read(self):
        if self._mode == "screen":
            return self._read_screen()
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.flip(frame, 1)

    def _read_screen(self):
        sct = self._mss.grab(self._monitor)
        frame = np.array(sct)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        # Downscale to max 1280px wide for detection performance
        h, w = frame.shape[:2]
        if w > 1280:
            scale = 1280 / w
            frame = cv2.resize(frame, (1280, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return frame  # no flip — screen is not mirrored

    def release(self):
        if self._mode == "screen":
            if self._mss:
                self._mss.close()
        else:
            self.cap.release()

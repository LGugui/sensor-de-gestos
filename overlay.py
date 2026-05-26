import time
import cv2


class Overlay:
    def __init__(self):
        self.active = True
        self._flash = 0
        self._fps_hist = []
        self._last_t = time.time()
        self._win = "Sensor de Gestos"

    def trigger_flash(self):
        self._flash = 3

    def _fps(self):
        now = time.time()
        dt = max(now - self._last_t, 0.001)
        self._last_t = now
        self._fps_hist.append(1.0 / dt)
        if len(self._fps_hist) > 10:
            self._fps_hist.pop(0)
        return sum(self._fps_hist) / len(self._fps_hist)

    def draw(self, frame, mode, hand_detected):
        fps = self._fps()

        if self._flash > 0:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 8)
            self._flash -= 1

        cor = (0, 255, 0) if hand_detected else (0, 0, 255)
        h = frame.shape[0]

        cv2.putText(frame, f"Modo: {mode}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2)
        cv2.putText(
            frame,
            "MAO DETECTADA" if hand_detected else "SEM MAO",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            cor,
            2,
        )
        cv2.putText(
            frame,
            "Q=sair | O=overlay | R=recarregar config",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 180, 180),
            1,
        )
        return frame

    def show(self, frame):
        cv2.imshow(self._win, frame)

    def toggle(self):
        self.active = not self.active
        if not self.active:
            cv2.destroyWindow(self._win)

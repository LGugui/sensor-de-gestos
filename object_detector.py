import queue
import threading

import cv2


class ObjectDetector:
    """YOLOv8n object detection running in a background daemon thread.
    Frames are submitted non-blocking — if the thread is busy the frame
    is dropped so hand tracking never stalls."""

    def __init__(self, model="yolov8n.pt", conf=0.4):
        from ultralytics import YOLO
        self._model = YOLO(model)
        self._conf = conf
        self._queue = queue.Queue(maxsize=1)
        self._results = None
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        try:
            self._queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # drop frame — thread busy

    def get_results(self):
        with self._lock:
            return self._results

    def _run(self):
        while True:
            frame = self._queue.get()
            try:
                res = self._model(frame, verbose=False, conf=self._conf)[0]
                with self._lock:
                    self._results = res
            except Exception:
                pass  # model failure on a single frame is non-fatal

    def draw(self, frame, results):
        if results is None:
            return frame
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = results.names[int(box.cls)]
            conf = float(box.conf)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1,
            )
        return frame

"""
Gesture data collector for TF training.

Run: python gesture_collector.py

Keys:
  0-8   : select gesture class
  SPACE : toggle recording
  C     : clear all samples for current class
  Q     : quit
"""
import csv
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(__file__))
import config as cfg_module
from camera import Camera
from detector import HandDetector
from tf_gesture import GESTURE_CLASSES, normalize_lm

DATA_DIR = os.path.join(os.path.dirname(__file__), "gesture_data")
DATA_FILE = os.path.join(DATA_DIR, "gestures.csv")

_COLLECT_CFG = {
    "camera_index": 0,
    "detection_confidence": 0.7,
    "tracking_confidence": 0.5,
    "max_hands": 1,
    "flip_handedness": True,
}


def _load_counts():
    counts = {cls: 0 for cls in GESTURE_CLASSES}
    if not os.path.exists(DATA_FILE):
        return counts
    with open(DATA_FILE, "r", newline="") as f:
        for row in csv.reader(f):
            if row and row[0] in counts:
                counts[row[0]] += 1
    return counts


def _clear_class(target_class):
    if not os.path.exists(DATA_FILE):
        return 0
    rows = []
    removed = 0
    with open(DATA_FILE, "r", newline="") as f:
        for row in csv.reader(f):
            if row and row[0] == target_class:
                removed += 1
            else:
                rows.append(row)
    with open(DATA_FILE, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    return removed


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    cam = Camera(_COLLECT_CFG["camera_index"])
    detector = HandDetector(_COLLECT_CFG)
    counts = _load_counts()

    current_class = 0
    recording = False
    status_msg = ""
    status_t = 0.0

    def flash(msg):
        nonlocal status_msg, status_t
        status_msg = msg
        status_t = time.time()

    print("Gesture Collector — pressione 0-8 para selecionar, ESPACO para gravar, Q para sair")

    with open(DATA_FILE, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)

        while True:
            frame = cam.read()
            if frame is None:
                break

            frame, hands = detector.find_hands(frame, draw=True)
            h, w = frame.shape[:2]

            lm = hands[0][0] if hands else None

            if recording and lm is not None:
                features = normalize_lm(lm)
                writer.writerow([GESTURE_CLASSES[current_class]] + features)
                counts[GESTURE_CLASSES[current_class]] += 1

            # ── Overlay ──────────────────────────────────────────────────
            cls_name = GESTURE_CLASSES[current_class]
            rec_color = (0, 80, 255) if recording else (0, 220, 255)

            # Recording indicator
            if recording:
                cv2.circle(frame, (w - 20, 20), 10, (0, 0, 255), -1)
                cv2.putText(frame, "GRAVANDO", (w - 100, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(frame, f"Classe: [{current_class}] {cls_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2)
            cv2.putText(frame, f"Amostras: {counts[cls_name]}",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            if lm is None:
                cv2.putText(frame, "Sem mao detectada", (10, 86),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 80, 255), 1)

            # Class list (right side)
            for i, cls in enumerate(GESTURE_CLASSES):
                marker = ">" if i == current_class else " "
                ok = counts[cls] >= 200
                col = (0, 255, 100) if ok else (180, 180, 180) if counts[cls] > 0 else (80, 80, 80)
                cv2.putText(frame, f"{marker}{i}:{cls} ({counts[cls]})",
                            (w - 235, 30 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

            # Status flash
            if status_msg and time.time() - status_t < 2.0:
                cv2.putText(frame, status_msg, (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(frame, "0-8:classe  ESPACO:gravar  C:limpar  Q:sair",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)

            cv2.imshow("Gesture Collector", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                recording = not recording
                if recording and lm is None:
                    recording = False
                    flash("Mostre a mao primeiro!")
            elif ord("0") <= key <= ord("8"):
                current_class = key - ord("0")
                recording = False
            elif key == ord("c"):
                removed = _clear_class(GESTURE_CLASSES[current_class])
                counts[GESTURE_CLASSES[current_class]] = 0
                flash(f"Removidas {removed} amostras de {GESTURE_CLASSES[current_class]}")

    cam.release()
    detector.close()
    cv2.destroyAllWindows()

    print("\nAmostras coletadas:")
    total = 0
    for cls in GESTURE_CLASSES:
        n = counts[cls]
        bar = "#" * (n // 10)
        status = "OK" if n >= 200 else f"precisam mais {200 - n}"
        print(f"  {cls:15s} {n:4d}  {bar:20s}  {status}")
        total += n
    print(f"\nTotal: {total} amostras em {DATA_FILE}")
    print("Para treinar: python gesture_trainer.py")


if __name__ == "__main__":
    main()

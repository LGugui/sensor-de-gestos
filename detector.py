import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_COLORS = {
    "Right": (0, 255, 0),
    "Left":  (0, 180, 255),
}

_FINGERTIPS = {4, 8, 12, 16, 20}
_PINCH_PAIR = (4, 8)
_TOUCH_VISUAL_DIST = 0.04
_VISIBILITY_THRESHOLD = 0.5

# Mesh geometry
_PALM_TRIS = [
    (0, 1, 5),
    (0, 5, 9),
    (0, 9, 13),
    (0, 13, 17),
    (5, 9, 13),
    (9, 13, 17),
]

_FINGER_CHAINS = [
    [1, 2, 3, 4],       # thumb
    [5, 6, 7, 8],       # index
    [9, 10, 11, 12],    # middle
    [13, 14, 15, 16],   # ring
    [17, 18, 19, 20],   # pinky
]

_FINGER_WIDTHS = [12, 9, 7, 5]   # base → tip

# Inter-finger webbing triangles (base of adjacent finger pairs)
_FINGER_WEBS = [
    (1, 2, 5),    # thumb → index web
    (5, 6, 9),    # index → middle web
    (9, 10, 13),  # middle → ring web
    (13, 14, 17), # ring → pinky web
]


def _finger_quad(pts, a, b, width):
    """Return numpy int32 array (4,2) — quad around bone a→b."""
    ax, ay = pts[a]
    bx, by = pts[b]
    dx, dy = bx - ax, by - ay
    length = math.sqrt(dx * dx + dy * dy) or 1
    px, py = -dy / length, dx / length
    half = width / 2
    return np.array([
        [int(ax + px * half), int(ay + py * half)],
        [int(ax - px * half), int(ay - py * half)],
        [int(bx - px * half), int(by - py * half)],
        [int(bx + px * half), int(by + py * half)],
    ], dtype=np.int32)


def _ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo de rastreamento de maos (~7MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo baixado.")


def _dim_color(color, factor):
    """Scale a BGR color tuple by factor [0,1]."""
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def _depth_style(z_avg):
    """Map z coordinate to (thickness, brightness_factor).
    z < 0 = in front of palm (closer to camera), z > 0 = behind.
    Range [-0.15, 0.15] → thickness [5, 1], brightness [1.0, 0.35].
    """
    t = max(0.0, min(1.0, (z_avg + 0.15) / 0.30))
    thickness = max(1, int(5 - t * 4))
    brightness = 1.0 - t * 0.65
    return thickness, brightness


def _is_landmark_visible(lm, idx, threshold=_VISIBILITY_THRESHOLD):
    vis = getattr(lm[idx], 'visibility', None)
    if vis is not None and vis >= 0:
        return vis >= threshold
    return lm[idx].z < 0.15


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
        self._vis_threshold = config.get("visibility_threshold", _VISIBILITY_THRESHOLD)

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
                raw_label = hd[0].category_name
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
        vis = [_is_landmark_visible(landmarks, i, self._vis_threshold)
               for i in range(len(landmarks))]

        # ── Bounding box ─────────────────────────────────────────────────
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 14
        bb_tl = (max(0, min(xs) - pad), max(0, min(ys) - pad))
        bb_br = (min(w - 1, max(xs) + pad), min(h - 1, max(ys) + pad))
        cv2.rectangle(frame, bb_tl, bb_br, _dim_color(color, 0.6), 1)
        cv2.putText(frame, label,
                    (bb_tl[0], bb_tl[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # ── Pinch check ───────────────────────────────────────────────────
        t4, t8 = landmarks[_PINCH_PAIR[0]], landmarks[_PINCH_PAIR[1]]
        pinch_dist = math.sqrt((t4.x - t8.x)**2 + (t4.y - t8.y)**2)
        pinch_contact = pinch_dist < _TOUCH_VISUAL_DIST

        # ── Mesh: fill all polygons into one overlay, then blend once ─────
        overlay = frame.copy()

        _GRAY_FILL = (30, 30, 30)
        _GRAY_WIRE = (70, 70, 70)

        # Palm triangles
        for tri in _PALM_TRIS:
            all_vis = all(vis[i] for i in tri)
            z_avg = sum(landmarks[i].z for i in tri) / 3
            _, brightness = _depth_style(z_avg)
            fill = _dim_color(color, brightness * 0.30) if all_vis else _GRAY_FILL
            tri_pts = np.array([pts[i] for i in tri], dtype=np.int32)
            cv2.fillPoly(overlay, [tri_pts], fill)

        # Finger webbing triangles
        for web in _FINGER_WEBS:
            all_vis = all(vis[i] for i in web)
            z_avg = sum(landmarks[i].z for i in web) / 3
            _, brightness = _depth_style(z_avg)
            fill = _dim_color(color, brightness * 0.22) if all_vis else _GRAY_FILL
            web_pts = np.array([pts[i] for i in web], dtype=np.int32)
            cv2.fillPoly(overlay, [web_pts], fill)

        # Finger tube quads
        for chain in _FINGER_CHAINS:
            for seg_idx in range(len(chain) - 1):
                a, b = chain[seg_idx], chain[seg_idx + 1]
                tube_w = _FINGER_WIDTHS[min(seg_idx, len(_FINGER_WIDTHS) - 1)]
                both_vis = vis[a] and vis[b]
                z_avg = (landmarks[a].z + landmarks[b].z) / 2
                _, brightness = _depth_style(z_avg)
                fill = _dim_color(color, brightness * 0.45) if both_vis else _GRAY_FILL
                quad = _finger_quad(pts, a, b, tube_w)
                cv2.fillPoly(overlay, [quad], fill)

        # One blend for all fills
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        # ── Wire edges drawn directly on frame (no blend) ─────────────────

        # Palm wire edges
        for tri in _PALM_TRIS:
            all_vis = all(vis[i] for i in tri)
            z_avg = sum(landmarks[i].z for i in tri) / 3
            _, brightness = _depth_style(z_avg)
            wire = _dim_color(color, brightness * 0.8) if all_vis else _GRAY_WIRE
            tri_pts = np.array([pts[i] for i in tri], dtype=np.int32)
            cv2.polylines(frame, [tri_pts], True, wire, 1)

        # Finger webbing wire edges
        for web in _FINGER_WEBS:
            all_vis = all(vis[i] for i in web)
            z_avg = sum(landmarks[i].z for i in web) / 3
            _, brightness = _depth_style(z_avg)
            wire = _dim_color(color, brightness * 0.5) if all_vis else _GRAY_WIRE
            web_pts = np.array([pts[i] for i in web], dtype=np.int32)
            cv2.polylines(frame, [web_pts], True, wire, 1)

        # Finger tube wire edges
        for chain in _FINGER_CHAINS:
            for seg_idx in range(len(chain) - 1):
                a, b = chain[seg_idx], chain[seg_idx + 1]
                tube_w = _FINGER_WIDTHS[min(seg_idx, len(_FINGER_WIDTHS) - 1)]
                both_vis = vis[a] and vis[b]
                z_avg = (landmarks[a].z + landmarks[b].z) / 2
                thickness, brightness = _depth_style(z_avg)
                wire = _dim_color(color, brightness) if both_vis else _GRAY_WIRE
                quad = _finger_quad(pts, a, b, tube_w)
                cv2.polylines(frame, [quad], True, wire, max(1, thickness - 1))

        # Pinch contact line (on top of mesh)
        if pinch_contact:
            cv2.line(frame, pts[4], pts[8], (0, 0, 255), 3)

        # ── Joint dots ───────────────────────────────────────────────────
        for i, (x, y) in enumerate(pts):
            is_tip = i in _FINGERTIPS
            is_pinch_tip = i in _PINCH_PAIR

            if not vis[i]:
                # Inactive — larger gray dot for tips, small for joints
                r = 10 if is_tip else 3
                cv2.circle(frame, (x, y), r, (45, 45, 45), -1)
                cv2.circle(frame, (x, y), r, (0, 0, 0), 1)
                continue

            z_val = landmarks[i].z
            _, brightness = _depth_style(z_val)

            if is_tip:
                radius = 10   # larger fingertip for better contact precision
                if is_pinch_tip and pinch_contact:
                    cv2.circle(frame, (x, y), radius + 5, (0, 0, 255), 2)
                    dot_color = (0, 0, 255)
                else:
                    ring_color = _dim_color((0, 200, 200), brightness)
                    cv2.circle(frame, (x, y), radius + 4, ring_color, 1)
                    dot_color = _dim_color((0, 255, 255), brightness)
            else:
                radius = 3
                dot_color = _dim_color((220, 220, 255) if label == "Right"
                                       else (255, 220, 180), brightness)

            cv2.circle(frame, (x, y), radius, dot_color, -1)
            cv2.circle(frame, (x, y), radius, (0, 0, 0), 1)

        # ── Finger distance indicators (thumb tip → each fingertip) ──────
        # Show normalized distance so user can calibrate gesture thresholds
        thumb_tip = landmarks[4]
        _FINGER_TIPS_IDX = [(8, "I"), (12, "M"), (16, "A"), (20, "P")]
        for tip_idx, finger_label in _FINGER_TIPS_IDX:
            if not (vis[4] and vis[tip_idx]):
                continue
            ft = landmarks[tip_idx]
            dist = math.sqrt((thumb_tip.x - ft.x)**2 + (thumb_tip.y - ft.y)**2)
            # Color: green when close (≤0.05), yellow when medium, white otherwise
            if dist <= 0.05:
                dcol = (0, 255, 80)
            elif dist <= 0.10:
                dcol = (0, 220, 255)
            else:
                dcol = (160, 160, 160)
            tx, ty = pts[tip_idx]
            cv2.putText(frame, f"{dist:.2f}", (tx + 12, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, dcol, 1)

        # Pinch midpoint
        if pinch_contact:
            mx = (pts[4][0] + pts[8][0]) // 2
            my = (pts[4][1] + pts[8][1]) // 2
            cv2.circle(frame, (mx, my), 6, (0, 0, 255), -1)
            cv2.putText(frame, "PINCH", (mx + 8, my + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    def close(self):
        self.detector.close()

import json
import os

DEFAULTS = {
    "camera_index": 0,
    "detection_confidence": 0.7,
    "tracking_confidence": 0.5,
    "max_hands": 1,
    "ema_alpha": 0.35,
    "cam_margin": 0.1,
    "dead_zone_px": 5,
    "pinch_close": 0.05,
    "pinch_open": 0.08,
    "precision_threshold_min": 0.08,
    "precision_threshold_max": 0.15,
    "precision_factor": 0.3,
    "scroll_speed": 3,
    "debounce_ms": 300,
    "menu_hold_seconds": 1.5,
}


def load(path="config.json"):
    if not os.path.exists(path):
        save(DEFAULTS, path)
        return DEFAULTS.copy()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return DEFAULTS.copy()
    result = DEFAULTS.copy()
    for key, default in DEFAULTS.items():
        val = data.get(key, default)
        if type(val) is type(default):
            result[key] = val
    return result


def save(cfg, path="config.json"):
    data = {"_comment": "Sensor de Gestos — edite com cuidado"}
    data.update({k: v for k, v in cfg.items() if not k.startswith("_")})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

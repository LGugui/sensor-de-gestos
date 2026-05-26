import time

HOLD_SECS = 2.0

# Clockwise from top-left
_CORNERS = [
    ("SUPERIOR ESQUERDO", 0, 0),
    ("SUPERIOR DIREITO",  1, 0),
    ("INFERIOR DIREITO",  1, 1),
    ("INFERIOR ESQUERDO", 0, 1),
]


class Calibrator:
    def __init__(self):
        self._step = 0
        self._hold_start = None
        self._samples = []
        self._data = []
        self.done = False
        self.result = None  # (min_x, max_x, min_y, max_y)

    @property
    def label(self):
        return _CORNERS[self._step][0] if self._step < len(_CORNERS) else ""

    @property
    def target_fx(self):
        return _CORNERS[self._step][1] if self._step < len(_CORNERS) else 0

    @property
    def target_fy(self):
        return _CORNERS[self._step][2] if self._step < len(_CORNERS) else 0

    @property
    def step(self):
        return self._step

    @property
    def total(self):
        return len(_CORNERS)

    def update(self, lm):
        """Returns hold progress [0,1], or None when all corners done."""
        if self.done or self._step >= len(_CORNERS):
            return None

        if self._hold_start is None:
            self._hold_start = time.time()

        self._samples.append((lm[9].x, lm[9].y))
        progress = min(1.0, (time.time() - self._hold_start) / HOLD_SECS)

        if progress >= 1.0:
            n = min(30, len(self._samples))
            recent = self._samples[-n:]
            avg_x = sum(s[0] for s in recent) / n
            avg_y = sum(s[1] for s in recent) / n
            self._data.append((avg_x, avg_y))
            self._step += 1
            self._hold_start = None
            self._samples = []
            if self._step >= len(_CORNERS):
                self._finalize()

        return progress

    def _finalize(self):
        xs = [d[0] for d in self._data]
        ys = [d[1] for d in self._data]
        pad_x = (max(xs) - min(xs)) * 0.03
        pad_y = (max(ys) - min(ys)) * 0.03
        self.result = (
            min(xs) - pad_x,
            max(xs) + pad_x,
            min(ys) - pad_y,
            max(ys) + pad_y,
        )
        self.done = True

import threading
import tkinter as tk


class DrawingOverlay:
    """Transparent fullscreen canvas for drawing with hand gestures.
    Runs tkinter in a daemon thread — call start() before use."""

    def __init__(self, color="#ff2222", line_width=3):
        self.color = color
        self.line_width = line_width
        self._lock = threading.Lock()
        self._pen_down = False
        self._x = 0
        self._y = 0
        self._prev_x = None
        self._prev_y = None
        self._clear_req = False
        self._root = None
        self._canvas = None
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def update(self, x, y, pen_down):
        with self._lock:
            self._x = x
            self._y = y
            self._pen_down = pen_down

    def clear(self):
        with self._lock:
            self._clear_req = True

    def _run(self):
        self._root = tk.Tk()
        self._root.attributes('-topmost', True)
        self._root.attributes('-fullscreen', True)
        self._root.wm_attributes('-transparentcolor', 'black')
        self._root.configure(bg='black')
        self._root.overrideredirect(True)

        self._canvas = tk.Canvas(
            self._root, bg='black',
            highlightthickness=0, cursor='none'
        )
        self._canvas.pack(fill='both', expand=True)

        self._root.after(16, self._tick)
        self._root.mainloop()

    def _tick(self):
        if not self._running:
            self._root.destroy()
            return

        with self._lock:
            x, y = self._x, self._y
            pen_down = self._pen_down
            clear = self._clear_req
            self._clear_req = False

        if clear:
            self._canvas.delete('all')
            self._prev_x = None
            self._prev_y = None

        if pen_down:
            if self._prev_x is not None:
                self._canvas.create_line(
                    self._prev_x, self._prev_y, x, y,
                    fill=self.color,
                    width=self.line_width,
                    smooth=True,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )
            self._prev_x = x
            self._prev_y = y
        else:
            self._prev_x = None
            self._prev_y = None

        self._root.after(16, self._tick)

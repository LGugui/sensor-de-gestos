import math
import os
import threading
import tkinter as tk
from datetime import datetime

SHAPES = ["circulo", "quadrado", "triangulo", "estrela"]

DRAW_PALETTE = ["#ff2222", "#00e5ff", "#00ff88", "#ffff00", "#ff8800", "#ff00ff"]


class DrawingOverlay:
    """Transparent fullscreen canvas. Created ONCE at startup, shown/hidden on demand."""

    def __init__(self, color="#ff2222", line_width=3):
        self.color = color
        self.line_width = line_width
        self._lock = threading.Lock()
        self._pen_down = False
        self._prev_pen_down = False
        self._x = 0
        self._y = 0
        self._prev_x = None
        self._prev_y = None
        self._clear_req = False
        self._undo_req = False
        self._save_req = None  # path string when save requested
        self._shape_queue = []
        self._visible = False
        self._root = None
        self._canvas = None
        self._ready = threading.Event()
        self._thread = None
        # stroke tracking for undo: list of lists of canvas IDs
        self._stroke_ids = []
        self._current_stroke = []

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=3.0)

    def stop(self):
        if self._root:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def show(self):
        with self._lock:
            self._visible = True
        if self._root:
            self._root.after(0, self._root.deiconify)

    def hide(self):
        with self._lock:
            self._visible = False
        if self._root:
            self._root.after(0, self._root.withdraw)

    def update(self, x, y, pen_down):
        with self._lock:
            self._x = x
            self._y = y
            self._pen_down = pen_down

    def clear(self):
        with self._lock:
            self._clear_req = True

    def undo(self):
        with self._lock:
            self._undo_req = True

    def set_color(self, color):
        with self._lock:
            self.color = color

    def save_png(self, path=None):
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(os.path.dirname(__file__), f"desenho_{ts}.png")
        with self._lock:
            self._save_req = path
        return path

    def stamp_shape(self, cx, cy, shape, radius):
        with self._lock:
            self._shape_queue.append((cx, cy, shape, int(radius)))

    def _run(self):
        self._root = tk.Tk()
        self._root.attributes('-topmost', True)
        self._root.attributes('-fullscreen', True)
        self._root.wm_attributes('-transparentcolor', 'black')
        self._root.configure(bg='black')
        self._root.overrideredirect(True)
        self._root.withdraw()

        self._canvas = tk.Canvas(
            self._root, bg='black',
            highlightthickness=0, cursor='none'
        )
        self._canvas.pack(fill='both', expand=True)

        self._ready.set()
        self._root.after(16, self._tick)
        self._root.mainloop()

    def _draw_shape(self, cx, cy, shape, radius, color):
        kw = dict(outline=color, width=self.line_width, fill="")
        item_id = None
        if shape == "circulo":
            item_id = self._canvas.create_oval(
                cx - radius, cy - radius, cx + radius, cy + radius, **kw)
        elif shape == "quadrado":
            item_id = self._canvas.create_rectangle(
                cx - radius, cy - radius, cx + radius, cy + radius, **kw)
        elif shape == "triangulo":
            h = int(radius * 0.866)
            pts = [cx, cy - radius, cx - radius, cy + h, cx + radius, cy + h]
            item_id = self._canvas.create_polygon(pts, **kw)
        elif shape == "estrela":
            pts = []
            for i in range(10):
                angle = math.radians(i * 36 - 90)
                r = radius if i % 2 == 0 else int(radius * 0.4)
                pts += [cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))]
            item_id = self._canvas.create_polygon(pts, **kw)
        if item_id is not None:
            self._stroke_ids.append([item_id])

    def _tick(self):
        with self._lock:
            x, y = self._x, self._y
            pen_down = self._pen_down
            clear = self._clear_req
            self._clear_req = False
            undo = self._undo_req
            self._undo_req = False
            save_path = self._save_req
            self._save_req = None
            shapes = self._shape_queue[:]
            self._shape_queue.clear()
            visible = self._visible
            color = self.color
            line_w = self.line_width

        if not visible:
            self._root.after(16, self._tick)
            return

        if clear:
            self._canvas.delete('all')
            self._prev_x = None
            self._prev_y = None
            self._stroke_ids.clear()
            self._current_stroke.clear()

        if undo and self._stroke_ids:
            ids = self._stroke_ids.pop()
            for iid in ids:
                self._canvas.delete(iid)
            self._prev_x = None
            self._prev_y = None

        for cx, cy, shape, radius in shapes:
            self._draw_shape(cx, cy, shape, radius, color)

        # Pen up → down: start new stroke
        if pen_down and not self._prev_pen_down:
            if self._current_stroke:
                self._stroke_ids.append(self._current_stroke)
            self._current_stroke = []

        if pen_down:
            if self._prev_x is not None:
                item_id = self._canvas.create_line(
                    self._prev_x, self._prev_y, x, y,
                    fill=color,
                    width=line_w,
                    smooth=True,
                    capstyle=tk.ROUND,
                    joinstyle=tk.ROUND,
                )
                self._current_stroke.append(item_id)
            self._prev_x = x
            self._prev_y = y
        else:
            if self._prev_pen_down and self._current_stroke:
                self._stroke_ids.append(self._current_stroke)
                self._current_stroke = []
            self._prev_x = None
            self._prev_y = None

        self._prev_pen_down = pen_down

        if save_path:
            self._do_save(save_path)

        self._root.after(16, self._tick)

    def _do_save(self, path):
        try:
            ps_path = path.replace(".png", ".ps")
            self._canvas.postscript(file=ps_path, colormode="color")
            try:
                from PIL import Image
                img = Image.open(ps_path)
                img.save(path)
                os.remove(ps_path)
            except ImportError:
                # PIL not available — keep .ps file
                print(f"PIL não instalado. Salvo como {ps_path}")
                return
            print(f"Canvas salvo: {path}")
        except Exception as e:
            print(f"Erro ao salvar canvas: {e}")

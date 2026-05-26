import tkinter as tk
from tkinter import ttk
import json
import os

BG = "#0f0f0f"
BG2 = "#1a1a1a"
ACCENT = "#00e5ff"
TEXT = "#e0e0e0"
TEXT_DIM = "#888888"
BTN_BG = "#1e1e1e"
BTN_HOVER = "#2a2a2a"
DANGER = "#ff4444"

RESULT_START    = "start"
RESULT_TUTORIAL = "tutorial"
RESULT_QUIT     = "quit"


class Launcher:
    def __init__(self, cfg_path="config.json"):
        self.cfg_path = cfg_path
        self.result = RESULT_QUIT
        self._cfg = self._load_cfg()
        self._root = None

    def _load_cfg(self):
        try:
            with open(self.cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_cfg(self):
        try:
            self._cfg["_comment"] = "Sensor de Gestos — edite com cuidado"
            with open(self.cfg_path, "w", encoding="utf-8") as f:
                json.dump(self._cfg, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def run(self):
        self._root = tk.Tk()
        self._root.title("Sensor de Gestos")
        self._root.configure(bg=BG)
        self._root.resizable(False, False)
        self._root.geometry("420x580")
        self._root.eval('tk::PlaceWindow . center')

        self._build_ui()
        self._root.mainloop()
        return self.result

    def _build_ui(self):
        r = self._root

        # Header
        hdr = tk.Frame(r, bg=ACCENT, height=4)
        hdr.pack(fill="x")

        tk.Label(r, text="✋", font=("Segoe UI Emoji", 40), bg=BG, fg=ACCENT).pack(pady=(24, 4))
        tk.Label(r, text="SENSOR DE GESTOS", font=("Segoe UI", 18, "bold"), bg=BG, fg=TEXT).pack()
        tk.Label(r, text="Controle seu PC com as mãos", font=("Segoe UI", 10), bg=BG, fg=TEXT_DIM).pack(pady=(2, 20))

        # Main buttons
        self._btn(r, "▶  INICIAR", self._on_start, accent=True)
        self._btn(r, "📖  TUTORIAL", self._on_tutorial)

        tk.Frame(r, bg=BG2, height=1).pack(fill="x", padx=40, pady=14)

        # Settings section
        tk.Label(r, text="CONFIGURAÇÕES", font=("Segoe UI", 9, "bold"),
                 bg=BG, fg=TEXT_DIM).pack()

        cfg_frame = tk.Frame(r, bg=BG2, padx=20, pady=16)
        cfg_frame.pack(fill="x", padx=24, pady=8)

        self._slider(cfg_frame, "Sensibilidade", "ema_alpha", 0.1, 0.9, 0.05)
        self._slider(cfg_frame, "Velocidade Scroll", "scroll_speed", 1, 10, 1)
        self._slider(cfg_frame, "Tempo Menu (s)", "menu_hold_seconds", 0.5, 3.0, 0.1)
        self._hand_selector(cfg_frame)

        tk.Frame(r, bg=BG2, height=1).pack(fill="x", padx=40, pady=10)
        self._btn(r, "✕  SAIR", self._on_quit, danger=True)

        tk.Label(r, text="github.com/LGugui/sensor-de-gestos",
                 font=("Segoe UI", 8), bg=BG, fg=TEXT_DIM).pack(side="bottom", pady=8)

    def _btn(self, parent, text, cmd, accent=False, danger=False):
        bg = ACCENT if accent else (DANGER if danger else BTN_BG)
        fg = "#000000" if accent else TEXT
        b = tk.Button(
            parent, text=text, command=cmd,
            bg=bg, fg=fg, activebackground=BTN_HOVER, activeforeground=TEXT,
            font=("Segoe UI", 11, "bold" if accent else "normal"),
            relief="flat", cursor="hand2",
            width=26, pady=10,
        )
        b.pack(pady=4)
        if not accent and not danger:
            b.bind("<Enter>", lambda e: b.configure(bg=BTN_HOVER))
            b.bind("<Leave>", lambda e: b.configure(bg=BTN_BG))

    def _slider(self, parent, label, key, from_, to, resolution):
        val = self._cfg.get(key, from_)
        var = tk.DoubleVar(value=val)

        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=3)
        tk.Label(row, text=label, bg=BG2, fg=TEXT, font=("Segoe UI", 9), width=18, anchor="w").pack(side="left")
        lbl = tk.Label(row, text=f"{val:.2g}", bg=BG2, fg=ACCENT, font=("Segoe UI", 9), width=5)
        lbl.pack(side="right")

        def on_change(v):
            rounded = round(float(v) / resolution) * resolution
            self._cfg[key] = round(rounded, 3)
            lbl.configure(text=f"{rounded:.2g}")
            self._save_cfg()

        s = ttk.Scale(row, from_=from_, to=to, variable=var,
                      orient="horizontal", length=140, command=on_change)
        s.pack(side="right", padx=6)

    def _hand_selector(self, parent):
        val = self._cfg.get("dominant_hand", "Right")
        var = tk.StringVar(value=val)
        row = tk.Frame(parent, bg=BG2)
        row.pack(fill="x", pady=4)
        tk.Label(row, text="Mão dominante", bg=BG2, fg=TEXT, font=("Segoe UI", 9), width=18, anchor="w").pack(side="left")
        for opt in ("Right", "Left"):
            tk.Radiobutton(
                row, text=opt, variable=var, value=opt,
                bg=BG2, fg=TEXT, selectcolor=BG,
                activebackground=BG2, font=("Segoe UI", 9),
                command=lambda: (self._cfg.update({"dominant_hand": var.get()}), self._save_cfg())
            ).pack(side="left", padx=4)

    def _on_start(self):
        self.result = RESULT_START
        self._root.destroy()

    def _on_tutorial(self):
        self.result = RESULT_TUTORIAL
        self._root.destroy()

    def _on_quit(self):
        self.result = RESULT_QUIT
        self._root.destroy()

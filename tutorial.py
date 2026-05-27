import cv2


_STEPS = [
    {
        "title": "BEM-VINDO!",
        "subtitle": "Sensor de Gestos — Tutorial",
        "desc": [
            "Vamos aprender os gestos basicos.",
            "Posicione sua mao na frente da camera.",
            "Pressione ESPACO para avancar.",
        ],
        "landmark_hint": None,
        "color": (0, 220, 255),
    },
    {
        "title": "MOVER O CURSOR",
        "subtitle": "Gesto: Indicador esticado",
        "desc": [
            "Estique o DEDO INDICADOR (mao direita).",
            "O cursor segue a ponta do dedo.",
            "Mova devagar para mais precisao.",
        ],
        "landmark_hint": 8,
        "color": (0, 255, 120),
    },
    {
        "title": "CLICAR",
        "subtitle": "Gesto: Pinch polegar + indicador",
        "desc": [
            "Aproxime POLEGAR e INDICADOR.",
            "= Clique esquerdo.",
            "Polegar + MEDIO = clique direito.",
        ],
        "landmark_hint": 8,
        "color": (255, 180, 0),
    },
    {
        "title": "PRECISAO",
        "subtitle": "Gesto: Semi-pinch",
        "desc": [
            "Aproxime polegar e indicador",
            "na METADE do caminho (sem fechar).",
            "Cursor fica 3x mais lento.",
        ],
        "landmark_hint": 8,
        "color": (180, 100, 255),
    },
    {
        "title": "SCROLL",
        "subtitle": "Gesto: Indicador + Medio eretos",
        "desc": [
            "Estique INDICADOR e MEDIO juntos.",
            "Mova a mao para CIMA = scroll up.",
            "Mova para BAIXO = scroll down.",
        ],
        "landmark_hint": 12,
        "color": (0, 200, 255),
    },
    {
        "title": "MENU",
        "subtitle": "Gesto: Ambas as palmas abertas",
        "desc": [
            "Abra AMBAS as maos completamente.",
            "Mantenha por 1.5 segundos.",
            "O menu de opcoes aparece.",
        ],
        "landmark_hint": None,
        "color": (0, 255, 200),
    },
    {
        "title": "TECLADO VIRTUAL",
        "subtitle": "Gesto: Punho com mao esquerda",
        "desc": [
            "Feche a MAO ESQUERDA (punho).",
            "Teclado QWERTY aparece na tela.",
            "Hover + pinch para digitar.",
        ],
        "landmark_hint": None,
        "color": (255, 120, 0),
    },
    {
        "title": "MODO DESENHO",
        "subtitle": "Gesto: So o anelar levantado (4s)",
        "desc": [
            "Levante so o ANELAR (4o dedo),",
            "segure 4 segundos. Barra enche.",
            "Repita anelar 4s para sair do modo.",
        ],
        "landmark_hint": 16,
        "color": (255, 60, 120),
    },
    {
        "title": "PRONTO!",
        "subtitle": "Tutorial concluido",
        "desc": [
            "Voce conhece todos os gestos.",
            "Pressione ESPACO para comecar.",
            "Pressione ? no app para ver resumo.",
        ],
        "landmark_hint": None,
        "color": (0, 220, 255),
    },
]


class Tutorial:
    def __init__(self):
        self.step = 0
        self.done = False
        self.total = len(_STEPS)

    def advance(self):
        self.step += 1
        if self.step >= self.total:
            self.done = True

    def back(self):
        self.step = max(0, self.step - 1)

    def draw(self, frame, landmarks=None):
        s = _STEPS[self.step]
        h, w = frame.shape[:2]
        color = s["color"]

        # Dim frame
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        # Highlight landmark
        if landmarks and s["landmark_hint"] is not None:
            idx = s["landmark_hint"]
            lx = int(landmarks[idx].x * w)
            ly = int(landmarks[idx].y * h)
            cv2.circle(frame, (lx, ly), 18, color, -1)
            cv2.circle(frame, (lx, ly), 24, color, 2)

        # Panel
        px, py, pw, ph = 30, 30, w - 60, h - 60
        panel = frame.copy()
        cv2.rectangle(panel, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
        cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), color, 2)

        # Step counter
        cv2.putText(frame, f"PASSO {self.step + 1}/{self.total}",
                    (px + 14, py + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

        # Title
        cv2.putText(frame, s["title"],
                    (px + 14, py + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        # Subtitle
        cv2.putText(frame, s["subtitle"],
                    (px + 14, py + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Divider
        cv2.line(frame, (px + 14, py + 112), (px + pw - 14, py + 112), color, 1)

        # Description lines
        for i, line in enumerate(s["desc"]):
            cv2.putText(frame, f"  {line}",
                        (px + 14, py + 145 + i * 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 1)

        # Progress bar
        bar_y = py + ph - 40
        bar_w = pw - 28
        filled = int(bar_w * (self.step + 1) / self.total)
        cv2.rectangle(frame, (px + 14, bar_y), (px + 14 + bar_w, bar_y + 8), (40, 40, 40), -1)
        cv2.rectangle(frame, (px + 14, bar_y), (px + 14 + filled, bar_y + 8), color, -1)

        # Navigation hint
        nav = "[ ESPACO ] Proximo    [ B ] Voltar    [ Q ] Sair"
        cv2.putText(frame, nav,
                    (px + 14, py + ph - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        return frame


def run_tutorial(cam, detector):
    """Run tutorial loop. Returns when done or user quits."""
    from camera import Camera
    tut = Tutorial()
    win = "Tutorial — Sensor de Gestos"

    while True:
        frame = cam.read()
        if frame is None:
            break

        frame, hands = detector.find_hands(frame, draw=True)
        landmarks = hands[0][0] if hands else None

        frame = tut.draw(frame, landmarks)
        cv2.imshow(win, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            tut.advance()
            if tut.done:
                break
        elif key == ord("b"):
            tut.back()

    cv2.destroyWindow(win)

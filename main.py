import cv2
from pynput.mouse import Button

import config as cfg_module
from camera import Camera
from controller import MouseController
from detector import HandDetector
from drawing_overlay import DrawingOverlay
from gestures import GestureDetector
from keyboard_overlay import VirtualKeyboard
from mapper import CoordMapper
from menu import GestureMenu
from overlay import Overlay

MODE_NORMAL    = "NAVEGACAO"
MODE_PRECISION = "PRECISAO"
MODE_SCROLL    = "SCROLL"
MODE_MENU      = "MENU"
MODE_KEYBOARD  = "TECLADO"
MODE_DRAW      = "DESENHO"


def _rebuild(cfg):
    return (
        HandDetector(cfg),
        CoordMapper(cfg),
        MouseController(cfg),
        GestureDetector(cfg),
    )


def _split_hands(hands, dominant):
    cursor = next((lm for lm, lbl in hands if lbl == dominant), None)
    control = next((lm for lm, lbl in hands if lbl != dominant), None)
    return cursor, control


def main():
    cfg = cfg_module.load()
    cam = Camera(cfg["camera_index"])
    detector, mapper, mouse, gestures = _rebuild(cfg)
    overlay = Overlay()
    menu = GestureMenu()
    keyboard = VirtualKeyboard()
    drawing = DrawingOverlay(color="#ff2222", line_width=3)

    cursor_x, cursor_y = mapper.screen_w // 2, mapper.screen_h // 2
    scroll_mode = False
    draw_mode = False
    mode = MODE_NORMAL

    print("Sensor de Gestos v3 — Dual Hand + Modo Desenho")
    print("  Q=sair | O=overlay | R=recarregar config")
    print()
    print("Gestos:")
    print("  Mao dominante (cursor): mover / pinch=clique / scroll")
    print("  Ambas palmas abertas 1.5s: menu")
    print("  Punho mao controle: teclado virtual (toggle)")
    print("  V sign mao dominante 1s: modo desenho (toggle)")
    print("  No desenho: pinch=caneta | punho mao controle=limpar")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            frame, hands = detector.find_hands(frame, draw=overlay.active)
            dominant = cfg["dominant_hand"]
            cursor_lm, control_lm = _split_hands(hands, dominant)
            hand_detected = cursor_lm is not None
            palm_progress = 0.0
            victory_progress = 0.0

            # ── DRAW mode ────────────────────────────────────────────────
            if draw_mode:
                mode = MODE_DRAW
                if hand_detected:
                    # V sign again → exit draw
                    victory_progress = gestures.detect_victory(cursor_lm)
                    if victory_progress >= 1.0:
                        draw_mode = False
                        drawing.stop()
                        mode = MODE_NORMAL
                    else:
                        # pinch = pen down
                        pen_down = (gestures._pinching_left or
                                    gestures._dist(cursor_lm[4], cursor_lm[8]) < gestures.pinch_close)
                        # move cursor normally, also update drawing canvas
                        nx, ny = mapper.map(cursor_lm, precision=False)
                        cursor_x, cursor_y = nx, ny
                        drawing.update(cursor_x, cursor_y, pen_down)
                        mouse.move(cursor_x, cursor_y)

                        # control hand fist = clear canvas
                        if control_lm and gestures.detect_fist(control_lm):
                            drawing.clear()
                else:
                    drawing.update(cursor_x, cursor_y, False)

            # ── KEYBOARD mode ─────────────────────────────────────────────
            elif keyboard.open:
                mode = MODE_KEYBOARD
                if hand_detected:
                    keyboard.update_hover(
                        frame.shape[1], frame.shape[0],
                        cursor_x, cursor_y,
                        mapper.screen_w, mapper.screen_h,
                    )
                    pinch = gestures.detect_pinch(cursor_lm)
                    if pinch == "left":
                        keyboard.press_hovered()
                        if overlay.active:
                            overlay.trigger_flash()
                    if control_lm and gestures.detect_fist(control_lm):
                        keyboard.close()
                        mode = MODE_NORMAL
                    nx, ny = mapper.map(cursor_lm, precision=False)
                    cursor_x, cursor_y = nx, ny
                    mouse.move(cursor_x, cursor_y)
                else:
                    keyboard.close()
                    mode = MODE_NORMAL

            # ── MENU mode ─────────────────────────────────────────────────
            elif menu.open:
                mode = MODE_MENU
                if hand_detected:
                    menu.update_hover(
                        frame.shape[1], frame.shape[0],
                        cursor_x, cursor_y,
                        mapper.screen_w, mapper.screen_h,
                    )
                    pinch = gestures.detect_pinch(cursor_lm)
                    if pinch == "left":
                        result = menu.select(cfg)
                        if result == "exit":
                            break
                        detector.close()
                        detector, mapper, mouse, gestures = _rebuild(cfg)
                    nx, ny = mapper.map(cursor_lm, precision=False)
                    cursor_x, cursor_y = nx, ny
                    mouse.move(cursor_x, cursor_y)
                else:
                    menu.close()
                    mode = MODE_NORMAL

            # ── NORMAL mode ───────────────────────────────────────────────
            else:
                mode = MODE_NORMAL

                if hand_detected:
                    # V sign → draw mode
                    victory_progress = gestures.detect_victory(cursor_lm)
                    if victory_progress >= 1.0:
                        draw_mode = True
                        drawing.start()
                        gestures.reset_pinch()
                        mode = MODE_DRAW

                    # Dual palm → menu
                    elif cursor_lm and control_lm:
                        palm_progress = gestures.detect_dual_palm(cursor_lm, control_lm)
                        if palm_progress >= 1.0:
                            menu.toggle()
                            gestures.reset_pinch()
                            mode = MODE_MENU

                    # Control fist → keyboard
                    elif control_lm and gestures.detect_fist(control_lm):
                        keyboard.toggle()
                        mode = MODE_KEYBOARD if keyboard.open else MODE_NORMAL

                    if not draw_mode and not menu.open and not keyboard.open:
                        # Single-hand palm fallback
                        if not control_lm:
                            palm_progress = gestures.detect_open_palm(cursor_lm)
                            if palm_progress >= 1.0:
                                menu.toggle()
                                gestures.reset_pinch()
                                mode = MODE_MENU

                        if not menu.open:
                            if gestures.detect_scroll_mode(cursor_lm):
                                mode = MODE_SCROLL
                                scroll_mode = True
                                delta = gestures.get_scroll_delta(cursor_lm)
                                if delta:
                                    mouse.scroll(delta)
                            else:
                                if scroll_mode:
                                    gestures.reset_scroll()
                                    scroll_mode = False
                                precision = gestures.detect_precision(cursor_lm)
                                if precision:
                                    mode = MODE_PRECISION
                                pinch = gestures.detect_pinch(cursor_lm)
                                if pinch == "left":
                                    clicked = mouse.click(Button.left)
                                    if clicked and overlay.active:
                                        overlay.trigger_flash()
                                elif pinch == "right":
                                    mouse.click(Button.right)
                                else:
                                    nx, ny = mapper.map(cursor_lm, precision=precision)
                                    cursor_x, cursor_y = nx, ny
                                    mouse.move(cursor_x, cursor_y)
                else:
                    if scroll_mode:
                        gestures.reset_scroll()
                        scroll_mode = False

            # ── Render ───────────────────────────────────────────────────
            if overlay.active:
                if keyboard.open:
                    frame = keyboard.draw(frame)

                # Victory/draw progress bar
                prog = victory_progress if victory_progress > 0 else palm_progress
                frame = overlay.draw(
                    frame, mode, hand_detected,
                    palm_progress=prog,
                    menu=menu if not keyboard.open else None,
                    cursor_pos=(cursor_x, cursor_y),
                )
                overlay.show(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                overlay.toggle()
            elif key == ord("r"):
                cfg = cfg_module.load()
                detector.close()
                detector, mapper, mouse, gestures = _rebuild(cfg)
                print("Config recarregada.")

    finally:
        if draw_mode:
            drawing.stop()
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()

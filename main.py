import cv2
from pynput.mouse import Button

import config as cfg_module
from camera import Camera
from controller import MouseController
from detector import HandDetector
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


def _rebuild(cfg):
    return (
        HandDetector(cfg),
        CoordMapper(cfg),
        MouseController(cfg),
        GestureDetector(cfg),
    )


def _split_hands(hands, dominant):
    """Return (cursor_lm, control_lm) from list of (lm, label)."""
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

    cursor_x, cursor_y = mapper.screen_w // 2, mapper.screen_h // 2
    scroll_mode = False
    mode = MODE_NORMAL

    print("Sensor de Gestos v2 — Dual Hand")
    print("  Q=sair | O=overlay | R=recarregar config")
    print()
    print("Gestos:")
    print("  Mao dominante (cursor): mover/pinch/scroll como antes")
    print("  Ambas as maos abertas 1.5s: menu")
    print("  Punho na mao de controle: teclado virtual (toggle)")
    print("  Se so uma mao: palma aberta 1.5s abre menu")

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

            # ── KEYBOARD mode ────────────────────────────────────────────
            if keyboard.open:
                mode = MODE_KEYBOARD
                if hand_detected:
                    keyboard.update_hover(
                        frame.shape[1], frame.shape[0],
                        cursor_x, cursor_y,
                        mapper.screen_w, mapper.screen_h,
                    )
                    pinch = gestures.detect_pinch(cursor_lm)
                    if pinch == "left":
                        closed = keyboard.press_hovered()
                        if not closed and overlay.active:
                            overlay.trigger_flash()

                    # control hand fist → close keyboard
                    if control_lm and gestures.detect_fist(control_lm):
                        keyboard.close()
                        mode = MODE_NORMAL

                    # still move cursor for hover
                    nx, ny = mapper.map(cursor_lm, precision=False)
                    cursor_x, cursor_y = nx, ny
                    mouse.move(cursor_x, cursor_y)

            # ── MENU mode ────────────────────────────────────────────────
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

            # ── NORMAL mode ──────────────────────────────────────────────
            else:
                mode = MODE_NORMAL

                # Two-hand dual-palm → menu
                if cursor_lm and control_lm:
                    palm_progress = gestures.detect_dual_palm(cursor_lm, control_lm)
                    if palm_progress >= 1.0:
                        menu.toggle()
                        gestures.reset_pinch()
                        mode = MODE_MENU

                # Control hand fist → toggle keyboard
                if control_lm and not menu.open:
                    if gestures.detect_fist(control_lm):
                        keyboard.toggle()
                        mode = MODE_KEYBOARD if keyboard.open else MODE_NORMAL

                if hand_detected and not menu.open and not keyboard.open:
                    # Single-hand palm fallback (no control hand present)
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
                            if delta != 0:
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

                elif not hand_detected:
                    if scroll_mode:
                        gestures.reset_scroll()
                        scroll_mode = False

            # ── Render ───────────────────────────────────────────────────
            if overlay.active:
                if keyboard.open:
                    frame = keyboard.draw(frame)
                frame = overlay.draw(
                    frame, mode, hand_detected,
                    palm_progress=palm_progress,
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
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()

import cv2
from pynput.mouse import Button

import config as cfg_module
from camera import Camera
from controller import MouseController
from detector import HandDetector
from gestures import GestureDetector
from mapper import CoordMapper
from menu import GestureMenu
from overlay import Overlay

MODE_NORMAL = "NAVEGACAO"
MODE_PRECISION = "PRECISAO"
MODE_SCROLL = "SCROLL"
MODE_MENU = "MENU"


def _rebuild(cfg):
    return (
        HandDetector(cfg),
        CoordMapper(cfg),
        MouseController(cfg),
        GestureDetector(cfg),
    )


def main():
    cfg = cfg_module.load()
    cam = Camera(cfg["camera_index"])
    detector, mapper, mouse, gestures = _rebuild(cfg)
    overlay = Overlay()
    menu = GestureMenu()

    cursor_x, cursor_y = mapper.screen_w // 2, mapper.screen_h // 2
    scroll_mode = False
    mode = MODE_NORMAL

    print("Sensor de Gestos iniciado.")
    print("  Q=sair | O=overlay | R=recarregar | Palma aberta 1.5s=menu")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            frame, landmarks = detector.find_hands(frame, draw=overlay.active)
            hand_detected = landmarks is not None
            palm_progress = 0.0

            if menu.open:
                mode = MODE_MENU
                if hand_detected:
                    menu.update_hover(
                        frame.shape[1], frame.shape[0],
                        cursor_x, cursor_y,
                        mapper.screen_w, mapper.screen_h,
                    )
                    pinch = gestures.detect_pinch(landmarks)
                    if pinch == "left":
                        result = menu.select(cfg)
                        if result == "exit":
                            break
                        # rebuild with updated config
                        detector.close()
                        detector, mapper, mouse, gestures = _rebuild(cfg)
                    # still move cursor so user can hover items
                    if not gestures.detect_scroll_mode(landmarks):
                        nx, ny = mapper.map(landmarks, precision=False)
                        cursor_x, cursor_y = nx, ny
                        mouse.move(cursor_x, cursor_y)
                # palm again or no hand timeout → close menu
                if not hand_detected:
                    menu.close()
                    mode = MODE_NORMAL

            else:
                mode = MODE_NORMAL
                if hand_detected:
                    # Check for menu trigger (open palm hold)
                    palm_progress = gestures.detect_open_palm(landmarks)
                    if palm_progress >= 1.0:
                        menu.toggle()
                        gestures.reset_pinch()
                        mode = MODE_MENU
                    elif gestures.detect_scroll_mode(landmarks):
                        mode = MODE_SCROLL
                        scroll_mode = True
                        delta = gestures.get_scroll_delta(landmarks)
                        if delta != 0:
                            mouse.scroll(delta)
                    else:
                        if scroll_mode:
                            gestures.reset_scroll()
                            scroll_mode = False

                        precision = gestures.detect_precision(landmarks)
                        if precision:
                            mode = MODE_PRECISION

                        pinch = gestures.detect_pinch(landmarks)
                        if pinch == "left":
                            clicked = mouse.click(Button.left)
                            if clicked and overlay.active:
                                overlay.trigger_flash()
                        elif pinch == "right":
                            mouse.click(Button.right)
                        else:
                            nx, ny = mapper.map(landmarks, precision=precision)
                            cursor_x, cursor_y = nx, ny
                            mouse.move(cursor_x, cursor_y)
                else:
                    if scroll_mode:
                        gestures.reset_scroll()
                        scroll_mode = False

            if overlay.active:
                frame = overlay.draw(
                    frame, mode, hand_detected,
                    palm_progress=palm_progress,
                    menu=menu,
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

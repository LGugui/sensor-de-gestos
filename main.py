import cv2
from pynput.mouse import Button

import config as cfg_module
from camera import Camera
from controller import MouseController
from detector import HandDetector
from gestures import GestureDetector
from mapper import CoordMapper
from overlay import Overlay


def main():
    cfg = cfg_module.load()

    cam = Camera(cfg["camera_index"])
    detector = HandDetector(cfg)
    mapper = CoordMapper(cfg)
    mouse = MouseController(cfg)
    gestures = GestureDetector(cfg)
    overlay = Overlay()

    scroll_mode = False

    print("Sensor de Gestos iniciado.")
    print("  Q = sair | O = toggle overlay | R = recarregar config")
    print()
    print("Gestos:")
    print("  Indicador [8]          = mover cursor")
    print("  Pinch polegar+indicador = clique esquerdo")
    print("  Pinch polegar+medio    = clique direito")
    print("  Semi-pinch             = modo precisao (30% velocidade)")
    print("  Indicador+medio eretos = modo scroll (mover mao p/ cima/baixo)")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            frame, landmarks = detector.find_hands(frame, draw=overlay.active)
            hand_detected = landmarks is not None
            mode = "NAVEGACAO"

            if hand_detected:
                if gestures.detect_scroll_mode(landmarks):
                    mode = "SCROLL"
                    if not scroll_mode:
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
                        mode = "PRECISAO"

                    pinch = gestures.detect_pinch(landmarks)
                    if pinch == "left":
                        clicked = mouse.click(Button.left)
                        if clicked and overlay.active:
                            overlay.trigger_flash()
                    elif pinch == "right":
                        mouse.click(Button.right)
                    else:
                        x, y = mapper.map(landmarks, precision=precision)
                        mouse.move(x, y)
            else:
                if scroll_mode:
                    gestures.reset_scroll()
                    scroll_mode = False

            if overlay.active:
                frame = overlay.draw(frame, mode, hand_detected)
                overlay.show(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                overlay.toggle()
            elif key == ord("r"):
                cfg = cfg_module.load()
                detector.close()
                detector = HandDetector(cfg)
                mapper = CoordMapper(cfg)
                mouse = MouseController(cfg)
                gestures = GestureDetector(cfg)
                print("Config recarregada.")

    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()

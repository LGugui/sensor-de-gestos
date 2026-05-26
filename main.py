import time

import cv2
from pynput.mouse import Button

import config as cfg_module
from calibration import Calibrator
from camera import Camera
from controller import MouseController
from detector import HandDetector
from drawing_overlay import DrawingOverlay, SHAPES, DRAW_PALETTE
from gestures import GestureDetector
from keyboard_overlay import VirtualKeyboard
from launcher import Launcher, RESULT_START, RESULT_TUTORIAL, RESULT_QUIT
from mapper import CoordMapper
from menu import GestureMenu
from overlay import Overlay
from tutorial import run_tutorial

MODE_NORMAL    = "NAVEGACAO"
MODE_PRECISION = "PRECISAO"
MODE_SCROLL    = "SCROLL"
MODE_MENU      = "MENU"
MODE_KEYBOARD  = "TECLADO"
MODE_DRAW      = "DESENHO"

_HAND_LOST_GRACE = 0.4  # seconds before keyboard/menu auto-close on hand loss


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
    # Launcher
    launcher = Launcher()
    launcher_result = launcher.run()
    if launcher_result == RESULT_QUIT:
        return

    cfg = cfg_module.load()
    cam = Camera(cfg["camera_index"])
    detector, mapper, mouse, gestures = _rebuild(cfg)

    # Load calibration if available; otherwise auto-start calibration
    saved_cal = cfg_module.load_calibration()
    if saved_cal:
        mapper.calibrate(*saved_cal)
        calibrator = None
    else:
        calibrator = Calibrator()

    if launcher_result == RESULT_TUTORIAL:
        run_tutorial(cam, detector)
    overlay = Overlay()
    menu = GestureMenu()
    keyboard = VirtualKeyboard()
    drawing = DrawingOverlay(color="#ff2222", line_width=3)
    drawing.start()

    cursor_x, cursor_y = mapper.screen_w // 2, mapper.screen_h // 2
    scroll_mode = False
    draw_mode = False
    shape_idx = 0
    color_idx = 0
    mode = MODE_NORMAL
    prev_hand_detected = False
    _dragging = False
    _hand_lost_t = None

    print("Sensor de Gestos v3 — Dual Hand + Modo Desenho")
    print("  Q=sair | O=overlay | R=reload | D=debug | K=teclado | B=pincel")
    print()
    print("Gestos:")
    print("  Mao dominante (cursor): mover / pinch=clique / scroll")
    print("  Ambas palmas abertas 1.5s: menu")
    print("  Punho mao controle: teclado virtual (toggle)")
    print("  Anelar mao dominante 4s: modo desenho (toggle)")
    print("  No desenho: pinch=caneta | punho mao controle=limpar")
    print("  Teclas no modo desenho: C=cor | Z=desfazer | E=salvar PNG")

    try:
        while True:
            frame = cam.read()
            if frame is None:
                break

            if mapper._ar_bounds is None and not mapper._cal:
                mapper.set_frame_size(frame.shape[1], frame.shape[0])

            frame, hands = detector.find_hands(frame, draw=overlay.active)
            dominant = cfg["dominant_hand"]
            cursor_lm, control_lm = _split_hands(hands, dominant)
            hand_detected = cursor_lm is not None
            if prev_hand_detected and not hand_detected:
                mapper.reset_position()
                if _dragging:
                    mouse.release(Button.left)
                    _dragging = False
            prev_hand_detected = hand_detected
            palm_progress = 0.0
            victory_progress = 0.0

            # ── CALIBRATION mode (priority) ──────────────────────────────
            if calibrator is not None and not calibrator.done:
                cal_progress = calibrator.update(cursor_lm) if hand_detected else 0.0
                if calibrator.done:
                    cfg_module.save_calibration(*calibrator.result)
                    mapper.calibrate(*calibrator.result)
                    calibrator = None
                else:
                    frame = overlay.draw_calibration(
                        frame, calibrator, hand_detected, cal_progress or 0.0)
                    overlay.show(frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("f"):
                        calibrator = None  # skip without saving
                    continue

            # ── DRAW mode ────────────────────────────────────────────────
            if draw_mode:
                mode = f"{MODE_DRAW}:{SHAPES[shape_idx]}"
                if hand_detected:
                    _hand_lost_t = None
                    # Ring again → exit draw
                    victory_progress = gestures.detect_ring_hold(cursor_lm)
                    if victory_progress >= 1.0:
                        draw_mode = False
                        drawing.hide()
                        gestures.reset_victory()
                        mode = MODE_NORMAL
                    else:
                        # pinch = pen down
                        pen_down = (gestures.is_pinching or
                                    gestures._dist(cursor_lm[4], cursor_lm[8]) < gestures.pinch_close)
                        nx, ny = mapper.map_draw(cursor_lm, precision=False)
                        cursor_x, cursor_y = nx, ny
                        drawing.update(cursor_x, cursor_y, pen_down)
                        mouse.move(cursor_x, cursor_y)

                        if control_lm:
                            if gestures.detect_fist(control_lm):
                                drawing.clear()
                            touch_fired, t_mx, t_my, t_span = gestures.detect_hand_touch(
                                cursor_lm, control_lm)
                            if touch_fired:
                                radius = max(30, min(350, int(t_span * mapper.screen_w * 0.5)))
                                drawing.stamp_shape(
                                    int(t_mx * mapper.screen_w),
                                    int(t_my * mapper.screen_h),
                                    SHAPES[shape_idx], radius,
                                )
                else:
                    drawing.update(cursor_x, cursor_y, False)

            # ── KEYBOARD mode ─────────────────────────────────────────────
            elif keyboard.open:
                mode = MODE_KEYBOARD
                if hand_detected:
                    _hand_lost_t = None
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
                    if _hand_lost_t is None:
                        _hand_lost_t = time.time()
                    elif time.time() - _hand_lost_t > _HAND_LOST_GRACE:
                        keyboard.close()
                        mode = MODE_NORMAL
                        _hand_lost_t = None

            # ── MENU mode ─────────────────────────────────────────────────
            elif menu.open:
                mode = MODE_MENU
                if hand_detected:
                    _hand_lost_t = None
                    menu.update_hover(
                        frame.shape[1], frame.shape[0],
                        cursor_x, cursor_y,
                        mapper.screen_w, mapper.screen_h,
                    )
                    pinch = gestures.detect_pinch(cursor_lm)
                    if pinch == "left":
                        menu_action = menu.select(cfg)
                        if menu_action == "exit":
                            break
                        detector.close()
                        detector, mapper, mouse, gestures = _rebuild(cfg)
                    nx, ny = mapper.map(cursor_lm, precision=False)
                    cursor_x, cursor_y = nx, ny
                    mouse.move(cursor_x, cursor_y)
                else:
                    if _hand_lost_t is None:
                        _hand_lost_t = time.time()
                    elif time.time() - _hand_lost_t > _HAND_LOST_GRACE:
                        menu.close()
                        mode = MODE_NORMAL
                        _hand_lost_t = None

            # ── NORMAL mode ───────────────────────────────────────────────
            else:
                mode = MODE_NORMAL
                _hand_lost_t = None

                if hand_detected:
                    # Ring hold → draw mode
                    victory_progress = gestures.detect_ring_hold(cursor_lm)
                    if victory_progress >= 1.0:
                        if _dragging:
                            mouse.release(Button.left)
                            _dragging = False
                        draw_mode = True
                        drawing.show()
                        gestures.reset_pinch()
                        gestures.reset_victory()
                        mode = MODE_DRAW

                    # Dual palm → menu
                    elif cursor_lm and control_lm:
                        palm_progress = gestures.detect_dual_palm(cursor_lm, control_lm)
                        if palm_progress >= 1.0:
                            if _dragging:
                                mouse.release(Button.left)
                                _dragging = False
                            menu.toggle()
                            gestures.reset_pinch()
                            mode = MODE_MENU

                    # Control fist OR dominant-hand pinky alone → keyboard
                    elif ((control_lm and gestures.detect_fist(control_lm))
                          or gestures.detect_pinky_only(cursor_lm)):
                        if _dragging:
                            mouse.release(Button.left)
                            _dragging = False
                        keyboard.toggle()
                        mode = MODE_KEYBOARD if keyboard.open else MODE_NORMAL

                    if not draw_mode and not menu.open and not keyboard.open:
                        # Single-hand palm fallback
                        if not control_lm:
                            palm_progress = gestures.detect_open_palm(cursor_lm)
                            if palm_progress >= 1.0:
                                if _dragging:
                                    mouse.release(Button.left)
                                    _dragging = False
                                menu.toggle()
                                gestures.reset_pinch()
                                mode = MODE_MENU

                        depth = gestures.estimate_depth(cursor_lm)

                        if not menu.open:
                            if gestures.detect_scroll_mode(cursor_lm):
                                mode = MODE_SCROLL
                                scroll_mode = True
                                delta_y = gestures.get_scroll_delta(cursor_lm)
                                delta_z = gestures.get_depth_scroll_delta(cursor_lm)
                                delta = delta_y + delta_z
                                if delta:
                                    mouse.scroll(delta)
                            else:
                                if scroll_mode:
                                    gestures.reset_scroll()
                                    gestures.reset_depth()
                                    scroll_mode = False
                                precision = gestures.detect_precision(cursor_lm)
                                if precision:
                                    mode = MODE_PRECISION

                                # Double-click (check before single pinch)
                                if gestures.detect_double_click(cursor_lm):
                                    mouse.double_click(Button.left)
                                    if overlay.active:
                                        overlay.trigger_flash()
                                else:
                                    pinch = gestures.detect_pinch(cursor_lm)
                                    if pinch == "left":
                                        if not _dragging:
                                            mouse.press(Button.left)
                                            _dragging = True
                                            if overlay.active:
                                                overlay.trigger_flash()
                                    elif _dragging and not gestures.is_dragging:
                                        mouse.release(Button.left)
                                        _dragging = False
                                    elif pinch == "right":
                                        mouse.click(Button.right)

                                nx, ny = mapper.map(cursor_lm, precision=precision, depth=depth)
                                cursor_x, cursor_y = nx, ny
                                mouse.move(cursor_x, cursor_y)
                else:
                    if scroll_mode:
                        gestures.reset_scroll()
                        gestures.reset_depth()
                        scroll_mode = False

            # ── Render ───────────────────────────────────────────────────
            if overlay.active:
                if keyboard.open:
                    frame = keyboard.draw(frame)

                prog = victory_progress if victory_progress > 0 else palm_progress
                frame = overlay.draw(
                    frame, mode, hand_detected,
                    palm_progress=prog,
                    menu=menu if not keyboard.open else None,
                    cursor_pos=(cursor_x, cursor_y),
                    debug_lm=cursor_lm,
                    gestures=gestures,
                    mapping_bounds=mapper.effective_bounds,
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
            elif key == ord("?") or key == ord("/"):
                overlay.show_cheatsheet = not overlay.show_cheatsheet
            elif key == ord("d"):
                overlay.show_debug = not overlay.show_debug
            elif key == ord("f"):
                calibrator = Calibrator()
                print("Recalibrando...")
            elif key == ord("k"):
                keyboard.toggle()
            elif key == ord("b"):
                if draw_mode:
                    draw_mode = False
                    drawing.hide()
                    gestures.reset_victory()
                else:
                    draw_mode = True
                    drawing.show()
                    gestures.reset_pinch()
                    gestures.reset_victory()
            elif key == ord("s"):
                shape_idx = (shape_idx + 1) % len(SHAPES)
                print(f"Forma: {SHAPES[shape_idx]}")
            elif key == ord("c"):
                color_idx = (color_idx + 1) % len(DRAW_PALETTE)
                drawing.set_color(DRAW_PALETTE[color_idx])
                print(f"Cor: {DRAW_PALETTE[color_idx]}")
            elif key == ord("z"):
                drawing.undo()
            elif key == ord("e"):
                saved = drawing.save_png()
                print(f"Salvando: {saved}")

    finally:
        if _dragging:
            mouse.release(Button.left)
        drawing.stop()
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()

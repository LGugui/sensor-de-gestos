import time

import cv2
from pynput.mouse import Button

import config as cfg_module
from calibration import Calibrator
from camera import Camera
from controller import MouseController
from detector import HandDetector
from drawing_overlay import DrawingOverlay, SHAPES, DRAW_PALETTE
from eye_gestures import EyeGestures
from face_detector import FaceDetector
from face_mapper import FaceMapper
from gaze_calibration import GazeCalibrator, load_gaze_calibration, save_gaze_calibration
from gestures import GestureDetector
from keyboard_overlay import VirtualKeyboard
from launcher import Launcher, RESULT_START, RESULT_TUTORIAL, RESULT_QUIT
from mapper import CoordMapper
from menu import GestureMenu
from object_detector import ObjectDetector
from overlay import Overlay
from palm_controls import PalmControls, ZONE_UP, ZONE_DOWN, ZONE_LEFT, ZONE_RIGHT, ZONE_CENTER
from tutorial import run_tutorial

MODE_NORMAL    = "NAVEGACAO"
MODE_PRECISION = "PRECISAO"
MODE_SCROLL    = "SCROLL"
MODE_MENU      = "MENU"
MODE_KEYBOARD  = "TECLADO"
MODE_DRAW      = "DESENHO"

_HAND_LOST_GRACE = 0.4  # seconds before keyboard/menu auto-close on hand loss

CURSOR_MAOS  = "MAOS"
CURSOR_ROSTO = "ROSTO"
CURSOR_OLHOS = "OLHOS"
_CURSOR_CYCLE = [CURSOR_MAOS, CURSOR_ROSTO, CURSOR_OLHOS]


def _rebuild(cfg):
    eye_on = cfg.get("eye_enabled", False)
    return (
        HandDetector(cfg),
        FaceDetector(cfg) if eye_on else None,
        CoordMapper(cfg),
        FaceMapper(cfg) if eye_on else None,
        MouseController(cfg),
        GestureDetector(cfg),
        EyeGestures(cfg) if eye_on else None,
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
    detector, face_det, mapper, face_mapper, mouse, gestures, eye_gestures = _rebuild(cfg)

    # Object detection (YOLO, background thread — disabled if not configured)
    obj_detector = None
    if cfg.get("object_detection_enabled", False):
        try:
            obj_detector = ObjectDetector(conf=cfg.get("object_detection_conf", 0.4))
        except Exception as e:
            print(f"[YOLO] Não carregado: {e}")

    palm_controls = PalmControls(cfg)

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
    keyboard = VirtualKeyboard(mapper.screen_w, mapper.screen_h)
    drawing = DrawingOverlay(color="#ff2222", line_width=3)
    drawing.start()

    cursor_x, cursor_y = mapper.screen_w // 2, mapper.screen_h // 2
    scroll_mode = False
    draw_mode = False
    _wink_scroll_active = False
    shape_idx = 0
    color_idx = 0
    mode = MODE_NORMAL
    prev_hand_detected = False
    _dragging = False
    _hand_lost_t = None
    touch_enabled = cfg.get("touch_enabled", True)

    # Eye / face tracking state
    cursor_source = CURSOR_MAOS
    gaze_calibrator = None
    _face_lost_t = None
    _gaze_quality = ""
    _gaze_quality_until = 0.0
    saved_gaze_cal = load_gaze_calibration()
    if saved_gaze_cal is not None and face_mapper is not None:
        face_mapper.set_gaze_calibration(saved_gaze_cal)

    print("Sensor de Gestos v3 — Dual Hand + Modo Desenho")
    print("  Q=sair | O=overlay | R=reload | D=debug | K=teclado | B=pincel")
    print()
    print("Gestos:")
    print("  Mao dominante (cursor): mover / pinch=clique / scroll")
    print("  Todos dedos juntos (tulipa) mao dominante: menu")
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

            # YOLO: submit frame non-blocking; draw latest results
            if obj_detector is not None:
                obj_detector.submit(frame)
                frame = obj_detector.draw(frame, obj_detector.get_results())

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
            watch_progress = 0.0
            eye_action = ""
            dwell_progress = 0.0

            # ── Face / eye detection ─────────────────────────────────────
            face_lm, face_blendshapes, face_matrix = None, None, None
            if face_det is not None:
                face_lm, face_blendshapes, face_matrix = face_det.find_face(frame)
                if face_matrix is not None:
                    face_mapper.process_head(face_matrix, face_lm, face_blendshapes)
                    _face_lost_t = None
                elif cursor_source != CURSOR_MAOS:
                    # Grace period: only reset if face missing > 0.4s (prevents jitter)
                    if _face_lost_t is None:
                        _face_lost_t = time.time()
                    elif time.time() - _face_lost_t > 0.4:
                        face_mapper.reset_position()
                        _face_lost_t = None

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

            # ── GAZE CALIBRATION mode ────────────────────────────────────
            if gaze_calibrator is not None and not gaze_calibrator.done:
                cal_progress = (gaze_calibrator.update(face_lm, face_mapper, face_blendshapes)
                                if face_lm is not None else 0.0)
                if gaze_calibrator.done:
                    save_gaze_calibration(gaze_calibrator.result,
                                          gaze_calibrator.quality,
                                          gaze_calibrator.residual_px)
                    face_mapper.set_gaze_calibration(gaze_calibrator.result,
                                                     gaze_calibrator.quality)
                    _gaze_quality = gaze_calibrator.quality
                    _gaze_quality_until = time.time() + 3.0
                    print(f"Calibracao concluida — qualidade: {gaze_calibrator.quality} "
                          f"({gaze_calibrator.residual_px:.0f}px residual)")
                    gaze_calibrator = None
                else:
                    frame = overlay.draw_calibration(
                        frame, gaze_calibrator, face_lm is not None, cal_progress or 0.0,
                        face_lm=face_lm)
                    overlay.show(frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("f"):
                        gaze_calibrator = None
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

                if cursor_source == CURSOR_ROSTO and face_mapper is not None:
                    # Head controls cursor; blink presses hovered key
                    if face_lm is not None:
                        cursor_x, cursor_y = face_mapper.map_head()
                        mouse.move(cursor_x, cursor_y)
                        eye_action = "TECLADO: MOVENDO"
                    else:
                        eye_action = "rosto: nao detectado"
                    keyboard.update_hover(
                        frame.shape[1], frame.shape[0],
                        cursor_x, cursor_y,
                        mapper.screen_w, mapper.screen_h,
                    )
                    if face_blendshapes is not None and eye_gestures is not None:
                        if eye_gestures.detect_blink(face_blendshapes):
                            keyboard.press_hovered()
                            if overlay.active:
                                overlay.trigger_flash()
                                overlay.trigger_ripple(cursor_x, cursor_y)
                            eye_action = "TECLADO: BLINK!"
                    if control_lm and gestures.detect_fist(control_lm):
                        keyboard.close()
                        mode = MODE_NORMAL

                else:
                    # Hand controls cursor (CURSOR_MAOS / CURSOR_OLHOS)
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
                                overlay.trigger_ripple(cursor_x, cursor_y)
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
                        hand_lm=cursor_lm,
                    )
                    pinch = gestures.detect_pinch(cursor_lm)
                    if pinch == "left":
                        menu_action = menu.select(cfg)
                        if menu_action == "exit":
                            break
                        if face_det is not None:
                            face_det.close()
                        detector.close()
                        detector, face_det, mapper, face_mapper, mouse, gestures, eye_gestures = _rebuild(cfg)
                        if saved_gaze_cal is not None and face_mapper is not None:
                            face_mapper.set_gaze_calibration(saved_gaze_cal)
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

                # ── Face / eye cursor sources ─────────────────────────────
                if cursor_source == CURSOR_ROSTO and face_mapper is not None:
                    if face_matrix is not None:
                        cursor_x, cursor_y = face_mapper.map_head()
                        mouse.move(cursor_x, cursor_y)
                        scroll_d = face_mapper.get_head_scroll_delta()
                        if scroll_d:
                            mouse.scroll(scroll_d)
                            eye_action = "SCROLL " + ("↑" if scroll_d > 0 else "↓")
                        else:
                            eye_action = "MOVENDO"
                    else:
                        eye_action = "rosto: nao detectado"

                elif cursor_source == CURSOR_OLHOS and face_mapper is not None:
                    if face_lm is not None:
                        saccade = False
                        if eye_gestures is not None:
                            ix, iy = face_mapper.extract_iris_normalized(face_lm)
                            saccade = eye_gestures.is_saccade(ix, iy)
                        gxy = face_mapper.map_gaze(face_lm, saccade=saccade)
                        if gxy is not None:
                            cursor_x, cursor_y = gxy
                            mouse.move(cursor_x, cursor_y)
                            eye_action = "MOVENDO"
                        else:
                            eye_action = "olhos: sem calibracao  G=calibrar"
                        if eye_gestures is not None and not saccade:
                            dwell_fired = eye_gestures.update_dwell(cursor_x, cursor_y)
                            dwell_progress = eye_gestures.dwell_progress
                            if dwell_fired:
                                mouse.click(Button.left)
                                if overlay.active:
                                    overlay.trigger_flash()
                                    overlay.trigger_ripple(cursor_x, cursor_y)
                                eye_action = "DWELL: clique!"
                            elif dwell_progress > 0.05:
                                eye_action = f"DWELL {int(dwell_progress * 100)}%"
                    else:
                        eye_action = "rosto: nao detectado"

                if hand_detected and not touch_enabled:
                    # Touch disabled: still move cursor (MAOS only), no gestures/clicks
                    if cursor_source == CURSOR_MAOS:
                        nx, ny = mapper.map(cursor_lm, precision=False)
                        cursor_x, cursor_y = nx, ny
                        mouse.move(cursor_x, cursor_y)

                elif hand_detected:
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

                    else:
                        # All-fingers-touch (tulip) hold → menu
                        watch_progress = gestures.detect_all_touch(cursor_lm)
                        palm_progress = watch_progress
                        if watch_progress >= 1.0:
                            if _dragging:
                                mouse.release(Button.left)
                                _dragging = False
                            menu.toggle()
                            gestures.reset_pinch()
                            gestures.reset_all_touch()
                            mode = MODE_MENU

                        # Watch tap (non-dominant index → dominant wrist) → immediate menu
                        if not menu.open and control_lm:
                            if gestures.detect_watch_tap(cursor_lm, control_lm):
                                if _dragging:
                                    mouse.release(Button.left)
                                    _dragging = False
                                menu.toggle()
                                gestures.reset_pinch()
                                gestures.reset_all_touch()
                                mode = MODE_MENU

                        # Control fist OR dominant-hand pinky alone → keyboard
                        if ((control_lm and gestures.detect_fist(control_lm))
                              or gestures.detect_pinky_only(cursor_lm)):
                            if _dragging:
                                mouse.release(Button.left)
                                _dragging = False
                            keyboard.toggle()
                            mode = MODE_KEYBOARD if keyboard.open else MODE_NORMAL

                    if not draw_mode and not menu.open and not keyboard.open:
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

                                if cursor_source == CURSOR_MAOS:
                                    # Double-click (check before single pinch)
                                    if gestures.detect_double_click(cursor_lm):
                                        mouse.double_click(Button.left)
                                        if overlay.active:
                                            overlay.trigger_flash()
                                            overlay.trigger_ripple(cursor_x, cursor_y)
                                    else:
                                        pinch = gestures.detect_pinch(cursor_lm)
                                        if pinch == "left":
                                            if not _dragging:
                                                mouse.press(Button.left)
                                                _dragging = True
                                                if overlay.active:
                                                    overlay.trigger_flash()
                                                    overlay.trigger_ripple(cursor_x, cursor_y)
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

            # ── Combined eye + hand commands ─────────────────────────────
            if cursor_source == CURSOR_OLHOS and cursor_lm is not None:
                if gestures.detect_pinch(cursor_lm) == "left":
                    mouse.click(Button.left)
                    if overlay.active:
                        overlay.trigger_flash()
                        overlay.trigger_ripple(cursor_x, cursor_y)
                    eye_action = "PINCH+GAZE: clique!"
                if face_blendshapes is not None and eye_gestures is not None:
                    if eye_gestures.detect_blink(face_blendshapes):
                        mouse.click(Button.left)
                        if overlay.active:
                            overlay.trigger_flash()
                            overlay.trigger_ripple(cursor_x, cursor_y)
                        eye_action = "BLINK: clique!"

            if cursor_source == CURSOR_ROSTO and not keyboard.open and not _wink_scroll_active:
                if cursor_lm is not None and gestures.detect_pinch(cursor_lm) == "left":
                    mouse.click(Button.left)
                    if overlay.active:
                        overlay.trigger_flash()
                        overlay.trigger_ripple(cursor_x, cursor_y)
                    eye_action = "PINCH+ROSTO: clique!"
                if face_blendshapes is not None and eye_gestures is not None:
                    if eye_gestures.detect_double_blink(face_blendshapes):
                        mouse.click(Button.left)
                        if overlay.active:
                            overlay.trigger_flash()
                            overlay.trigger_ripple(cursor_x, cursor_y)
                        eye_action = "PISCADA DUPLA: clique!"

            # ── Left wink sustained → scroll mode toggle ─────────────────
            if eye_gestures is not None and face_blendshapes is not None:
                if eye_gestures.detect_left_wink(face_blendshapes):
                    if _wink_scroll_active:
                        mouse.release(Button.middle)
                        _wink_scroll_active = False
                        eye_action = "SCROLL: DESATIVADO"
                    else:
                        mouse.press(Button.middle)
                        _wink_scroll_active = True
                        eye_action = "SCROLL: ATIVADO"
                    if overlay.active:
                        overlay.trigger_flash()
                elif not eye_action:
                    wp = eye_gestures.wink_progress
                    if wp > 0.05:
                        eye_action = f"PISCAR ESQ {int(wp * 100)}%"

            # ── Long eye close → keyboard toggle ─────────────────────────
            if eye_gestures is not None and face_blendshapes is not None:
                if eye_gestures.detect_long_close(face_blendshapes):
                    keyboard.toggle()
                    if keyboard.open and cursor_source != CURSOR_ROSTO and face_mapper is not None:
                        cursor_source = CURSOR_ROSTO
                        face_mapper.reset_position()
                    mode = MODE_KEYBOARD if keyboard.open else MODE_NORMAL
                    if overlay.active:
                        overlay.trigger_flash()
                    eye_action = "TECLADO: ABERTO" if keyboard.open else "TECLADO: FECHADO"
                elif not eye_action:
                    lcp = eye_gestures.long_close_progress
                    if lcp > 0.05:
                        eye_action = f"FECHAR OLHOS {int(lcp * 100)}%"

            # ── Palm D-pad (control hand always, regardless of mode) ─────
            if control_lm and cursor_lm:
                fw, fh = frame.shape[1], frame.shape[0]
                fired_zone = palm_controls.update(cursor_lm, control_lm, fw, fh)
                if fired_zone == ZONE_UP:
                    mouse.scroll(cfg.get("scroll_speed", 3) * 3)
                elif fired_zone == ZONE_DOWN:
                    mouse.scroll(-cfg.get("scroll_speed", 3) * 3)
                elif fired_zone == ZONE_LEFT:
                    try:
                        import pyautogui
                        pyautogui.hotkey("alt", "left")
                    except Exception:
                        pass
                elif fired_zone == ZONE_RIGHT:
                    try:
                        import pyautogui
                        pyautogui.hotkey("alt", "right")
                    except Exception:
                        pass
                elif fired_zone == ZONE_CENTER:
                    mouse.click(Button.middle)

            # ── Render ───────────────────────────────────────────────────
            if control_lm:
                frame = palm_controls.draw(frame, control_lm)

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
                    screen_size=(mapper.screen_w, mapper.screen_h),
                    touch_enabled=touch_enabled,
                    draw_color=drawing.color,
                    watch_progress=watch_progress,
                    cursor_source=cursor_source,
                    gaze_calibrated=(face_mapper is not None and face_mapper._gaze_cal is not None),
                    face_lm=face_lm,
                    face_blendshapes=face_blendshapes,
                    eye_action=eye_action,
                    dwell_progress=dwell_progress,
                    gaze_quality=_gaze_quality if time.time() < _gaze_quality_until else "",
                )
                overlay.show(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("o"):
                overlay.toggle()
            elif key == ord("t"):
                touch_enabled = not touch_enabled
                if _dragging and not touch_enabled:
                    mouse.release(Button.left)
                    _dragging = False
                    gestures.reset_pinch()
                print(f"Toques: {'ATIVADOS' if touch_enabled else 'DESATIVADOS'}")
            elif key == ord("r"):
                cfg = cfg_module.load()
                if face_det is not None:
                    face_det.close()
                detector.close()
                detector, face_det, mapper, face_mapper, mouse, gestures, eye_gestures = _rebuild(cfg)
                if saved_gaze_cal is not None and face_mapper is not None:
                    face_mapper.set_gaze_calibration(saved_gaze_cal)
                touch_enabled = cfg.get("touch_enabled", True)
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
                if draw_mode:
                    saved = drawing.save_png()
                    print(f"Salvando: {saved}")
                elif not cfg.get("eye_enabled", False):
                    print("E: eye_enabled=false no config.json — ative para usar ROSTO/OLHOS")
                elif cfg.get("eye_enabled", False):
                    idx = _CURSOR_CYCLE.index(cursor_source)
                    cursor_source = _CURSOR_CYCLE[(idx + 1) % len(_CURSOR_CYCLE)]
                    if face_mapper is not None:
                        face_mapper.reset_position()
                    if eye_gestures is not None:
                        eye_gestures._dwell_start = None
                    print(f"Fonte cursor: {cursor_source}")
            elif key == ord("g"):
                if cfg.get("eye_enabled", False) and face_det is not None:
                    gaze_calibrator = GazeCalibrator()
                    print("Calibracao de olhar iniciada — olhe para cada ponto e segure 2s")

    finally:
        if _wink_scroll_active:
            mouse.release(Button.middle)
        if _dragging:
            mouse.release(Button.left)
        drawing.stop()
        cam.release()
        if face_det is not None:
            face_det.close()
        detector.close()
        cv2.destroyAllWindows()
        print("Encerrado.")


if __name__ == "__main__":
    main()

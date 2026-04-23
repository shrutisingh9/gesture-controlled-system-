"""Main entry point for AI-based gesture controlled system."""

from __future__ import annotations

import sys
import time

import cv2

from gesture_system.action_controller import ActionController
from gesture_system.config import CAMERA, MODES, UI
from gesture_system.gesture_recognizer import GestureRecognizer
from gesture_system.hand_tracker import HandTracker


def validate_runtime() -> None:
    major, minor = sys.version_info.major, sys.version_info.minor
    if (major, minor) >= (3, 13):
        raise RuntimeError(
            "This project requires Python 3.10 or 3.11 for MediaPipe stability. "
            f"Detected Python {major}.{minor}. "
            "Create a 3.11 virtual environment and reinstall requirements."
        )


def draw_overlay(frame, active_mode: str, active_gesture: str, fps: float, is_running: bool) -> None:
    cv2.rectangle(frame, (10, 10), (430, 160), UI.panel_color, -1)
    cv2.putText(
        frame,
        f"Mode: {active_mode}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        UI.font_scale,
        UI.text_color,
        UI.text_thickness,
    )
    cv2.putText(
        frame,
        f"Gesture: {active_gesture}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        UI.font_scale,
        UI.text_color,
        UI.text_thickness,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        UI.font_scale,
        UI.text_color,
        UI.text_thickness,
    )
    status = "RUNNING" if is_running else "PAUSED"
    color = UI.text_color if is_running else UI.warning_color
    cv2.putText(
        frame,
        f"Status: {status}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def draw_canvas(frame, points: list[tuple[int, int]]) -> None:
    if len(points) < 2:
        return
    for i in range(1, len(points)):
        cv2.line(frame, points[i - 1], points[i], (255, 0, 255), 3)


def main() -> None:
    validate_runtime()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA.height)

    try:
        tracker = HandTracker(CAMERA.detection_confidence, CAMERA.tracking_confidence)
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize MediaPipe hand tracking. "
            "Please install a supported MediaPipe build in a Python 3.10/3.11 virtual environment."
        ) from exc
    recognizer = GestureRecognizer()
    actions = ActionController(CAMERA.width, CAMERA.height)

    running = True
    mode_index = 0
    active_mode = MODES[mode_index]
    active_gesture = "NONE"

    prev_time = time.time()

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        results = tracker.process(frame)
        hands = tracker.extract_landmarks(frame, results)

        if hands:
            dominant = next((h for h in hands if h["side"] == "Right"), hands[0])
            tracker.draw(frame, dominant)
            data = recognizer.recognize(dominant, active_mode)
            active_gesture = data["gesture"]

            if running:
                index_tip = data.get("index_tip")
                wrist = data.get("wrist")

                if active_gesture == "MOVE_CURSOR" and index_tip:
                    actions.move_cursor(index_tip)
                    actions.drag_release()
                elif active_gesture == "LEFT_CLICK":
                    actions.left_click()
                elif active_gesture == "RIGHT_CLICK":
                    actions.right_click()
                elif active_gesture == "DOUBLE_CLICK":
                    actions.double_click()
                elif active_gesture == "DRAG" and index_tip:
                    actions.drag(index_tip)
                elif active_gesture == "SCROLL" and index_tip:
                    actions.scroll(index_tip[1])
                elif active_gesture == "VOLUME":
                    actions.set_volume_by_distance(data.get("distance", 0))
                elif active_gesture == "BRIGHTNESS":
                    actions.set_brightness_by_y(data.get("hand_y", wrist[1] if wrist else 0))
                elif active_gesture == "PLAY_PAUSE":
                    actions.media_key("play/pause media")
                elif active_gesture == "MEDIA_NEXT":
                    actions.media_key("next track")
                elif active_gesture == "MEDIA_PREV":
                    actions.media_key("previous track")
                elif active_gesture == "SCREENSHOT":
                    actions.screenshot()
                elif active_gesture == "LOCK_SYSTEM":
                    actions.lock_system()
                elif active_gesture == "OPEN_APP":
                    actions.open_app("notepad")
                elif active_gesture == "MINIMIZE_WINDOW":
                    actions.minimize_active_window()
                elif active_gesture == "MAXIMIZE_RESTORE_WINDOW":
                    actions.maximize_restore_active_window()
                elif active_gesture == "CLOSE_APP":
                    actions.close_active_app()
                elif active_gesture == "DRAW" and index_tip:
                    actions.update_drawing(True, index_tip)
                elif active_gesture == "DRAW_IDLE":
                    actions.update_drawing(False, None)
                    actions.drag_release()
                else:
                    actions.drag_release()
        else:
            active_gesture = "NO_HAND"
            actions.drag_release()
            actions.last_scroll_y = None

        if active_mode == "DRAW":
            draw_canvas(frame, actions.drawing_points)

        now = time.time()
        fps = 1 / max(now - prev_time, 1e-6)
        prev_time = now
        draw_overlay(frame, active_mode, active_gesture, fps, running)

        cv2.putText(
            frame,
            "Keys: [s] start/stop  [m] mode switch  [c] clear draw  [q] quit",
            (10, CAMERA.height - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 255, 200),
            2,
        )
        cv2.imshow("AI Gesture Controlled System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            running = not running
            if not running:
                actions.drag_release()
        if key == ord("m"):
            mode_index = (mode_index + 1) % len(MODES)
            active_mode = MODES[mode_index]
            actions.drag_release()
        if key == ord("c") and active_mode == "DRAW":
            actions.clear_drawing()

    actions.drag_release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

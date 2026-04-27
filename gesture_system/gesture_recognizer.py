from __future__ import annotations

import time
from dataclasses import dataclass

from .config import THRESHOLDS
from .utils import euclidean

TIP_IDS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
PIP_IDS = {"index": 6, "middle": 10, "ring": 14, "pinky": 18}


@dataclass
class GestureState:
    name: str = "NONE"
    pinch_active: bool = False
    drag_active: bool = False
    drag_start_time: float = 0.0
    last_pinch_time: float = 0.0
    scroll_anchor_y: float | None = None
    swipe_anchor_x: float | None = None
    candidate_gesture: str = "NONE"
    candidate_count: int = 0


class GestureRecognizer:
    def __init__(self) -> None:
        # Stores timing/state needed for temporal gestures (drag, double-click, swipe).
        self.state = GestureState()

    def _stabilize(self, gesture: str) -> str:
        # Require repeated frames for non-instant gestures to reduce false triggers.
        if gesture == "NONE":
            self.state.candidate_gesture = "NONE"
            self.state.candidate_count = 0
            return "NONE"

        if gesture == self.state.candidate_gesture:
            self.state.candidate_count += 1
        else:
            self.state.candidate_gesture = gesture
            self.state.candidate_count = 1

        if self.state.candidate_count >= THRESHOLDS.gesture_confirm_frames:
            return gesture
        return "NONE"

    @staticmethod
    def finger_states(lm: list[tuple[int, int]]) -> dict[str, bool]:
        # Finger "up" logic based on tip vs knuckle geometry.
        thumb_up = lm[TIP_IDS["thumb"]][0] > lm[TIP_IDS["thumb"] - 1][0]
        return {
            "thumb": thumb_up,
            "index": lm[TIP_IDS["index"]][1] < lm[PIP_IDS["index"]][1],
            "middle": lm[TIP_IDS["middle"]][1] < lm[PIP_IDS["middle"]][1],
            "ring": lm[TIP_IDS["ring"]][1] < lm[PIP_IDS["ring"]][1],
            "pinky": lm[TIP_IDS["pinky"]][1] < lm[PIP_IDS["pinky"]][1],
        }

    def recognize(self, hand: dict, mode: str) -> dict:
        # Core rule-based recognizer.
        lm = hand["landmarks"]
        fingers = self.finger_states(lm)
        thumb = lm[TIP_IDS["thumb"]]
        index = lm[TIP_IDS["index"]]
        middle = lm[TIP_IDS["middle"]]
        wrist = lm[0]
        now = time.time()

        pinch_index = euclidean(thumb, index)
        pinch_middle = euclidean(thumb, middle)

        gesture = "NONE"
        payload = {"fingers": fingers, "index_tip": index, "wrist": wrist}

        if mode == "MOUSE":
            # Keep scroll as highest priority in mouse mode so it does not
            # get overridden by any utility gesture checks below.
            if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                payload["gesture"] = self._stabilize("SCROLL")
                return payload

            if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "MOVE_CURSOR"

            if pinch_index < THRESHOLDS.pinch_distance:
                # Pinch hold transitions into drag after hold duration.
                if not self.state.pinch_active:
                    self.state.drag_start_time = now
                    self.state.pinch_active = True
                elif now - self.state.drag_start_time >= THRESHOLDS.drag_hold_time_s:
                    gesture = "DRAG"
                    self.state.drag_active = True
            else:
                # Click is emitted on pinch release; this avoids accidental open while starting drag.
                if self.state.pinch_active:
                    if self.state.drag_active:
                        gesture = "NONE"
                    else:
                        # Click only on pinch release to avoid opening items while starting drag.
                        if now - self.state.last_pinch_time < THRESHOLDS.double_click_window_s:
                            gesture = "DOUBLE_CLICK"
                        else:
                            gesture = "LEFT_CLICK"
                        self.state.last_pinch_time = now
                self.state.pinch_active = False
                self.state.drag_active = False

            if pinch_middle < THRESHOLDS.pinch_distance:
                gesture = "RIGHT_CLICK"

        elif mode == "MEDIA":
            # Media mode uses distance/pose and horizontal wrist swipe.
            # Volume is controlled by thumb-index distance. Keep this lenient so it is easy to trigger.
            if (not fingers["middle"]) and (not fingers["ring"]) and (not fingers["pinky"]) and pinch_index < (
                THRESHOLDS.pinch_distance * 3.0
            ):
                gesture = "VOLUME"
                payload["distance"] = pinch_index
            elif fingers["index"] and fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
                gesture = "PLAY_PAUSE"
            elif fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
                gesture = "BRIGHTNESS"
                payload["hand_y"] = wrist[1]
            else:
                if self.state.swipe_anchor_x is None:
                    self.state.swipe_anchor_x = wrist[0]
                dx = wrist[0] - self.state.swipe_anchor_x
                if dx > THRESHOLDS.swipe_min_dx:
                    gesture = "MEDIA_NEXT"
                    self.state.swipe_anchor_x = wrist[0]
                elif dx < -THRESHOLDS.swipe_min_dx:
                    gesture = "MEDIA_PREV"
                    self.state.swipe_anchor_x = wrist[0]

        elif mode == "DRAW":
            # Drawing mode: index-only draws, fist pauses drawing.
            if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "DRAW"
            elif not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "DRAW_IDLE"

        # Global utility gestures (mode-independent).
        # if fingers["thumb"] and fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
        #     gesture = "SCREENSHOT"
        # if fingers["thumb"] and fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
        #     gesture = "LOCK_SYSTEM"
        if not fingers["thumb"] and fingers["index"] and fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            gesture = "MINIMIZE_WINDOW"
        if fingers["thumb"] and not fingers["index"] and not fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
            gesture = "MAXIMIZE_RESTORE_WINDOW"
        if fingers["thumb"] and not fingers["index"] and fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            gesture = "CLOSE_APP"

        # Click-type gestures are instant events; do not delay them with frame stabilization.
        if gesture in {"LEFT_CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"}:
            payload["gesture"] = gesture
            return payload

        payload["gesture"] = self._stabilize(gesture)
        return payload

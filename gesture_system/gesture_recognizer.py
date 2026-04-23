"""Rule-based gesture recognition from hand landmarks."""

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


class GestureRecognizer:
    def __init__(self) -> None:
        self.state = GestureState()

    @staticmethod
    def finger_states(lm: list[tuple[int, int]]) -> dict[str, bool]:
        thumb_up = lm[TIP_IDS["thumb"]][0] > lm[TIP_IDS["thumb"] - 1][0]
        return {
            "thumb": thumb_up,
            "index": lm[TIP_IDS["index"]][1] < lm[PIP_IDS["index"]][1],
            "middle": lm[TIP_IDS["middle"]][1] < lm[PIP_IDS["middle"]][1],
            "ring": lm[TIP_IDS["ring"]][1] < lm[PIP_IDS["ring"]][1],
            "pinky": lm[TIP_IDS["pinky"]][1] < lm[PIP_IDS["pinky"]][1],
        }

    def recognize(self, hand: dict, mode: str) -> dict:
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
            if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "MOVE_CURSOR"

            if pinch_index < THRESHOLDS.pinch_distance:
                if now - self.state.last_pinch_time < THRESHOLDS.double_click_window_s:
                    gesture = "DOUBLE_CLICK"
                else:
                    gesture = "LEFT_CLICK"
                self.state.last_pinch_time = now

                if not self.state.pinch_active:
                    self.state.drag_start_time = now
                    self.state.pinch_active = True
                elif now - self.state.drag_start_time >= THRESHOLDS.drag_hold_time_s:
                    gesture = "DRAG"
                    self.state.drag_active = True
            else:
                self.state.pinch_active = False
                self.state.drag_active = False

            if pinch_middle < THRESHOLDS.pinch_distance:
                gesture = "RIGHT_CLICK"

            if fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "SCROLL"

        elif mode == "MEDIA":
            if fingers["thumb"] and fingers["index"] and not fingers["middle"]:
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
            if fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "DRAW"
            elif not fingers["index"] and not fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
                gesture = "DRAW_IDLE"

        if fingers["thumb"] and fingers["index"] and fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            gesture = "LOCK_SYSTEM"
        if fingers["thumb"] and fingers["index"] and fingers["middle"] and not fingers["ring"] and not fingers["pinky"]:
            gesture = "SCREENSHOT"
        if not fingers["thumb"] and fingers["index"] and fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            gesture = "MINIMIZE_WINDOW"
        if fingers["thumb"] and not fingers["index"] and not fingers["middle"] and fingers["ring"] and not fingers["pinky"]:
            gesture = "MAXIMIZE_RESTORE_WINDOW"
        if not fingers["thumb"] and fingers["index"] and not fingers["middle"] and fingers["ring"] and fingers["pinky"]:
            gesture = "OPEN_APP"
        if fingers["thumb"] and not fingers["index"] and fingers["middle"] and not fingers["ring"] and fingers["pinky"]:
            gesture = "CLOSE_APP"

        payload["gesture"] = gesture
        return payload

"""Map recognized gestures to system automation actions."""

from __future__ import annotations

import ctypes
import os
import subprocess
import time

import numpy as np
import pyautogui

from .config import THRESHOLDS
from .utils import Cooldown, LowPassFilter, clamp

try:
    import keyboard
except Exception:
    keyboard = None

try:
    import screen_brightness_control as sbc
except Exception:
    sbc = None

try:
    from ctypes import POINTER, cast

    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
except Exception:
    AudioUtilities = None
    IAudioEndpointVolume = None
    CLSCTX_ALL = None
    cast = None
    POINTER = None


class ActionController:
    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.screen_w, self.screen_h = pyautogui.size()
        self.cursor_filter = LowPassFilter(alpha=0.35)
        self.click_cooldown = Cooldown(THRESHOLDS.click_cooldown_s)
        self.media_cooldown = Cooldown(0.6)
        self.window_cooldown = Cooldown(0.8)
        self.last_scroll_y = None
        self.volume_interface = self._init_volume()
        self.drawing_points: list[tuple[int, int]] = []
        self.is_dragging = False

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.01

    def _init_volume(self):
        if not AudioUtilities:
            return None
        try:
            device = AudioUtilities.GetSpeakers()
            interface = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            return None

    def move_cursor(self, index_tip: tuple[int, int]) -> None:
        x = np.interp(index_tip[0], [0, self.frame_width], [0, self.screen_w])
        y = np.interp(index_tip[1], [0, self.frame_height], [0, self.screen_h])
        sx, sy = self.cursor_filter.update(x, y)
        pyautogui.moveTo(int(sx), int(sy))

    def left_click(self) -> None:
        if self.click_cooldown.ready():
            pyautogui.click(button="left")

    def right_click(self) -> None:
        if self.click_cooldown.ready():
            pyautogui.click(button="right")

    def double_click(self) -> None:
        if self.click_cooldown.ready():
            pyautogui.doubleClick()

    def drag(self, index_tip: tuple[int, int]) -> None:
        self.move_cursor(index_tip)
        if not self.is_dragging:
            pyautogui.mouseDown(button="left")
            self.is_dragging = True

    def drag_release(self) -> None:
        if self.is_dragging:
            pyautogui.mouseUp(button="left")
            self.is_dragging = False

    def scroll(self, y: int) -> None:
        if self.last_scroll_y is None:
            self.last_scroll_y = y
            return
        dy = self.last_scroll_y - y
        self.last_scroll_y = y
        pyautogui.scroll(int(dy * THRESHOLDS.scroll_sensitivity))

    def set_volume_by_distance(self, distance: float) -> None:
        if not self.volume_interface:
            return
        distance = clamp(distance, 20.0, 200.0)
        level = np.interp(distance, [20.0, 200.0], [0.0, 1.0])
        self.volume_interface.SetMasterVolumeLevelScalar(float(level), None)

    def set_brightness_by_y(self, y: float) -> None:
        if not sbc:
            return
        percent = np.interp(y, [0, self.frame_height], [100, 10])
        try:
            sbc.set_brightness(int(clamp(percent, 10, 100)))
        except Exception:
            pass

    def media_key(self, key: str) -> None:
        if not keyboard:
            return
        if self.media_cooldown.ready():
            keyboard.send(key)

    def screenshot(self) -> None:
        stamp = int(time.time())
        path = os.path.abspath(f"screenshot_{stamp}.png")
        pyautogui.screenshot(path)

    def lock_system(self) -> None:
        ctypes.windll.user32.LockWorkStation()

    def open_app(self, app_name: str = "notepad") -> None:
        try:
            subprocess.Popen([app_name], shell=True)
        except Exception:
            pass

    def close_active_app(self) -> None:
        pyautogui.hotkey("alt", "f4")

    def minimize_active_window(self) -> None:
        if not self.window_cooldown.ready():
            return
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if hwnd:
                ctypes.windll.user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
        except Exception:
            pass

    def maximize_restore_active_window(self) -> None:
        if not self.window_cooldown.ready():
            return
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            if not hwnd:
                return
            is_maximized = bool(ctypes.windll.user32.IsZoomed(hwnd))
            if is_maximized:
                ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            else:
                ctypes.windll.user32.ShowWindow(hwnd, 3)  # SW_MAXIMIZE
        except Exception:
            pass

    def update_drawing(self, draw_active: bool, point: tuple[int, int] | None) -> None:
        if draw_active and point is not None:
            self.drawing_points.append(point)
        elif not draw_active and len(self.drawing_points) > 3000:
            self.drawing_points = self.drawing_points[-2000:]

    def clear_drawing(self) -> None:
        self.drawing_points.clear()

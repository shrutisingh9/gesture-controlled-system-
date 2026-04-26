"""Central configuration for gesture system."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    smoothing_factor: float = 0.7
    detection_confidence: float = 0.7
    tracking_confidence: float = 0.6
    low_light_enhancement: bool = False
    target_fps: int = 20
    process_every_n_frames: int = 2


@dataclass(frozen=True)
class GestureThresholds:
    pinch_distance: float = 35.0
    click_cooldown_s: float = 0.25
    double_click_window_s: float = 0.35
    drag_hold_time_s: float = 0.4
    mode_switch_cooldown_s: float = 0.8
    swipe_min_dx: float = 70.0
    scroll_sensitivity: float = 1.5
    sudden_movement_filter_px: float = 120.0
    gesture_confirm_frames: int = 3


@dataclass(frozen=True)
class UIConfig:
    font_scale: float = 0.8
    text_thickness: int = 2
    panel_color: tuple = (35, 35, 35)
    text_color: tuple = (0, 255, 0)
    warning_color: tuple = (0, 0, 255)


CAMERA = CameraConfig()
THRESHOLDS = GestureThresholds()
UI = UIConfig()

MODES = ["MOUSE", "MEDIA", "DRAW"]

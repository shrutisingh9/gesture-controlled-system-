"""Utility helpers for geometry, smoothing, and time."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass


def euclidean(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.dist(p1, p2)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class LowPassFilter:
    alpha: float
    value_x: float = 0.0
    value_y: float = 0.0
    initialized: bool = False

    def update(self, x: float, y: float) -> tuple[float, float]:
        if not self.initialized:
            self.value_x, self.value_y = x, y
            self.initialized = True
            return x, y

        self.value_x = self.alpha * x + (1 - self.alpha) * self.value_x
        self.value_y = self.alpha * y + (1 - self.alpha) * self.value_y
        return self.value_x, self.value_y


class Cooldown:
    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self._last = 0.0

    def ready(self) -> bool:
        now = time.time()
        if now - self._last >= self.cooldown_s:
            self._last = now
            return True
        return False

"""Turn hand landmarks into pointer actions. Pure logic: no camera, no display."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

# MediaPipe hand landmark indices.
WRIST, THUMB_TIP, INDEX_MCP, INDEX_TIP, MIDDLE_MCP, MIDDLE_TIP, PINKY_MCP = 0, 4, 5, 8, 9, 12, 17

LEFT, RIGHT = "left", "right"


class Pointer(Protocol):
    def move(self, x: float, y: float) -> None:
        """Move to a position given as fractions (0..1) of the screen."""

    def button(self, name: str, pressed: bool) -> None: ...


@dataclass(frozen=True)
class Hand:
    """What the controller needs to know about one tracked hand in one frame."""

    x: float  # palm centre, as fractions of the (mirrored) camera frame
    y: float
    pinch_index: float  # thumb-to-fingertip distance in hand lengths
    pinch_middle: float


def observe(points: Sequence[tuple[float, float]], aspect: float) -> Hand:
    """Summarise 21 normalised landmarks. `aspect` is frame width / height."""

    def dist(a: int, b: int) -> float:
        return math.hypot((points[a][0] - points[b][0]) * aspect, points[a][1] - points[b][1])

    # Palm length and palm width foreshorten under different tilts, so take
    # whichever is less squashed (a palm is about 1.5x as long as it is wide).
    scale = max(dist(WRIST, MIDDLE_MCP), 1.5 * dist(INDEX_MCP, PINKY_MCP), 1e-6)
    # The palm barely moves when the fingers pinch, so clicking does not drag
    # the cursor the way a fingertip anchor would.
    palm = (WRIST, INDEX_MCP, PINKY_MCP)
    return Hand(
        x=sum(points[i][0] for i in palm) / len(palm),
        y=sum(points[i][1] for i in palm) / len(palm),
        pinch_index=dist(THUMB_TIP, INDEX_TIP) / scale,
        pinch_middle=dist(THUMB_TIP, MIDDLE_TIP) / scale,
    )


class OneEuroFilter:
    """Low-pass filter whose cutoff rises with speed: steady at rest, responsive in motion."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: float | None = None
        self._dx = 0.0
        self._t = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: float, t: float) -> float:
        if self._x is None or t <= self._t:
            self._x, self._dx, self._t = x, 0.0, t
            return x
        dt = t - self._t
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx += a_d * ((x - self._x) / dt - self._dx)
        a = self._alpha(self.min_cutoff + self.beta * abs(self._dx), dt)
        self._x += a * (x - self._x)
        self._t = t
        return self._x


@dataclass
class Config:
    margin: float = 0.25  # border of the camera frame that maps beyond the screen edge
    pinch_press: float = 0.22  # pinch distance (hand lengths) that presses a button
    pinch_release: float = 0.33  # ...and the larger distance that releases it
    smoothing: float = 1.0  # higher = steadier but laggier cursor
    drag_threshold: float = 0.015  # screen fraction the hand must travel before a held pinch drags
    lost_grace: float = 0.25  # seconds a held button survives the hand not being detected


class Controller:
    """Feeds one hand observation per frame into pointer moves and button events."""

    def __init__(self, pointer: Pointer, config: Config | None = None):
        self.pointer = pointer
        self.config = config or Config()
        cutoff = 1.0 / max(self.config.smoothing, 1e-3)
        self._fx = OneEuroFilter(cutoff, beta=10.0)
        self._fy = OneEuroFilter(cutoff, beta=10.0)
        self.held: str | None = None
        self._press_pos = (0.0, 0.0)
        self._dragging = False
        self._last_seen = -math.inf

    def reset(self) -> None:
        """Release any held button and forget the cursor history."""
        if self.held:
            self.pointer.button(self.held, False)
            self.held = None
        self._fx.reset()
        self._fy.reset()

    def update(self, hand: Hand | None, t: float) -> None:
        cfg = self.config
        if hand is None:
            if t - self._last_seen > cfg.lost_grace:
                self.reset()
            return
        self._last_seen = t

        x = self._fx(self._to_screen(hand.x), t)
        y = self._fy(self._to_screen(hand.y), t)
        pinches = {LEFT: hand.pinch_index, RIGHT: hand.pinch_middle}

        if self.held:
            # A click must not smear into a drag: the cursor stays put while a
            # button is down until the hand has clearly moved away.
            if not self._dragging and math.dist((x, y), self._press_pos) >= cfg.drag_threshold:
                self._dragging = True
            if self._dragging:
                self.pointer.move(x, y)
            if pinches[self.held] > cfg.pinch_release:
                self.pointer.button(self.held, False)
                self.held = None
            return

        self.pointer.move(x, y)
        name = min(pinches, key=pinches.get)
        if pinches[name] < cfg.pinch_press:
            self.pointer.button(name, True)
            self.held, self._press_pos, self._dragging = name, (x, y), False

    def _to_screen(self, v: float) -> float:
        m = self.config.margin
        return min(max((v - m) / (1 - 2 * m), 0.0), 1.0)

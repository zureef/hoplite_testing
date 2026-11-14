"""Backends that drive the system pointer."""

from __future__ import annotations

import fcntl
import os
import struct

from .gestures import LEFT, RIGHT


class PointerError(RuntimeError):
    pass


# From <linux/input-event-codes.h> and <linux/uinput.h>.
_EV_SYN, _EV_KEY, _EV_ABS = 0, 1, 3
_SYN_REPORT, _ABS_X, _ABS_Y = 0, 0, 1
_BTN = {LEFT: 0x110, RIGHT: 0x111}
_BUS_VIRTUAL = 0x06
_UI_SET_EVBIT, _UI_SET_KEYBIT, _UI_SET_ABSBIT = 0x40045564, 0x40045565, 0x40045567
_UI_DEV_CREATE, _UI_DEV_DESTROY = 0x5501, 0x5502
_ABS_CNT = 64
# struct uinput_user_dev: name, input_id, ff_effects_max, then absmax/absmin/absfuzz/absflat.
_USER_DEV = f"80sHHHHI{4 * _ABS_CNT}i"
_EVENT = "llHHi"  # struct input_event


class UinputPointer:
    """A virtual absolute pointing device, like a VM's USB tablet.

    The kernel hands it to the compositor as ordinary hardware, so it works on
    Wayland as well as X11. Needs write access to /dev/uinput.
    """

    _MAX = 65535

    def __init__(self, path: str = "/dev/uinput"):
        try:
            self._fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            raise PointerError(
                f"Cannot open {path}: {e.strerror}. handmouse needs write access to it "
                "to move the pointer on Wayland; see 'Wayland setup' in README.md."
            ) from e
        for ev in (_EV_SYN, _EV_KEY, _EV_ABS):
            fcntl.ioctl(self._fd, _UI_SET_EVBIT, ev)
        for code in _BTN.values():
            fcntl.ioctl(self._fd, _UI_SET_KEYBIT, code)
        absmax = [0] * _ABS_CNT
        for axis in (_ABS_X, _ABS_Y):
            fcntl.ioctl(self._fd, _UI_SET_ABSBIT, axis)
            absmax[axis] = self._MAX
        os.write(
            self._fd,
            struct.pack(_USER_DEV, b"handmouse virtual pointer", _BUS_VIRTUAL, 0x1, 0x1, 1, 0, *absmax, *[0] * (3 * _ABS_CNT)),
        )
        fcntl.ioctl(self._fd, _UI_DEV_CREATE)

    def _emit(self, *events: tuple[int, int, int]) -> None:
        events += ((_EV_SYN, _SYN_REPORT, 0),)
        os.write(self._fd, b"".join(struct.pack(_EVENT, 0, 0, *e) for e in events))

    def move(self, x: float, y: float) -> None:
        self._emit((_EV_ABS, _ABS_X, round(x * self._MAX)), (_EV_ABS, _ABS_Y, round(y * self._MAX)))

    def button(self, name: str, pressed: bool) -> None:
        self._emit((_EV_KEY, _BTN[name], int(pressed)))

    def close(self) -> None:
        fcntl.ioctl(self._fd, _UI_DEV_DESTROY)
        os.close(self._fd)


class X11Pointer:
    """Synthetic input through the XTEST extension. Needs no setup, X11 sessions only."""

    _BUTTON = {LEFT: 1, RIGHT: 3}

    def __init__(self):
        from Xlib import X, display, error
        from Xlib.ext import xtest

        self._X, self._xtest = X, xtest
        try:
            self._display = display.Display()
        except (error.DisplayError, error.XauthError, OSError) as e:
            raise PointerError(f"Cannot connect to the X server: {e}") from e
        if not self._display.has_extension("XTEST"):
            raise PointerError("The X server does not support the XTEST extension.")
        geometry = self._display.screen().root.get_geometry()
        self._width, self._height = geometry.width, geometry.height

    def move(self, x: float, y: float) -> None:
        self._xtest.fake_input(
            self._display, self._X.MotionNotify, x=round(x * (self._width - 1)), y=round(y * (self._height - 1))
        )
        self._display.flush()

    def button(self, name: str, pressed: bool) -> None:
        event = self._X.ButtonPress if pressed else self._X.ButtonRelease
        self._xtest.fake_input(self._display, event, self._BUTTON[name])
        self._display.flush()

    def close(self) -> None:
        self._display.close()


def open_pointer(backend: str = "auto") -> UinputPointer | X11Pointer:
    if backend == "auto":
        # Under Wayland DISPLAY is usually set too (XWayland), but XTEST would
        # only reach X11 windows there.
        wayland = os.environ.get("WAYLAND_DISPLAY") or not os.environ.get("DISPLAY")
        backend = "uinput" if wayland else "x11"
    return UinputPointer() if backend == "uinput" else X11Pointer()

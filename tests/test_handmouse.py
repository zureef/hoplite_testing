import struct

import pytest

from handmouse import pointer
from handmouse.gestures import LEFT, RIGHT, Config, Controller, Hand, OneEuroFilter, observe


class FakePointer:
    def __init__(self):
        self.events = []

    def move(self, x, y):
        self.events.append(("move", x, y))

    def button(self, name, pressed):
        self.events.append(("button", name, pressed))

    @property
    def buttons(self):
        return [e[1:] for e in self.events if e[0] == "button"]

    @property
    def position(self):
        return next(e[1:] for e in reversed(self.events) if e[0] == "move")


OPEN = dict(pinch_index=1.0, pinch_middle=1.0)
PINCH = dict(pinch_index=0.1, pinch_middle=1.0)


def feed(controller, hands, start=0.0, fps=30):
    for i, hand in enumerate(hands):
        controller.update(hand, start + i / fps)
    return start + len(hands) / fps


def landmarks(thumb_tip, index_tip, middle_tip=(0.5, 0.2)):
    points = [(0.0, 0.0)] * 21
    points[0] = (0.5, 0.8)  # wrist
    points[5] = (0.44, 0.5)  # index knuckle
    points[9] = (0.5, 0.5)  # middle knuckle
    points[17] = (0.56, 0.5)  # pinky knuckle
    points[4], points[8], points[12] = thumb_tip, index_tip, middle_tip
    return points


def test_observe_measures_pinch_in_hand_lengths():
    # 4:3 frame; the hand is 0.3 frame heights long.
    touching = observe(landmarks((0.4, 0.4), (0.4, 0.43)), aspect=4 / 3)
    apart = observe(landmarks((0.3, 0.5), (0.45, 0.2)), aspect=4 / 3)
    assert touching.pinch_index == pytest.approx(0.1)
    assert touching.pinch_index < Config().pinch_press < Config().pinch_release < apart.pinch_index
    assert (touching.x, touching.y) == pytest.approx((0.5, 0.6))


def test_observe_is_independent_of_hand_distance_from_camera():
    near = landmarks((0.4, 0.4), (0.4, 0.46))
    far = [(x / 2, y / 2) for x, y in near]
    assert observe(near, 4 / 3).pinch_index == pytest.approx(observe(far, 4 / 3).pinch_index)


def test_filter_settles_on_a_steady_value_and_tracks_fast_motion():
    f = OneEuroFilter(min_cutoff=1.0, beta=10.0)
    noisy = [f(0.5 + (0.002 if i % 2 else -0.002), i / 30) for i in range(60)]
    assert max(noisy[30:]) - min(noisy[30:]) < 0.002  # jitter is damped
    for i in range(60, 75):
        out = f(1.0, i / 30)
    assert out > 0.95  # a real move gets through within half a second


def test_active_area_maps_to_whole_screen():
    p = FakePointer()
    c = Controller(p, Config(margin=0.25))
    c.update(Hand(0.25, 0.75, **OPEN), 0.0)
    assert p.position == pytest.approx((0.0, 1.0))
    c.reset()
    c.update(Hand(0.5, 0.1, **OPEN), 1.0)
    assert p.position == pytest.approx((0.5, 0.0))  # clamped at the screen edge


def test_pinch_and_release_is_one_click_at_a_fixed_spot():
    p = FakePointer()
    c = Controller(p)
    t = feed(c, [Hand(0.5, 0.5, **OPEN)] * 5)
    before = p.position
    # The hand wobbles slightly while pinching, as real hands do.
    t = feed(c, [Hand(0.5 + 0.002 * (i % 2), 0.5, **PINCH) for i in range(6)], t)
    assert p.position == before
    feed(c, [Hand(0.5, 0.5, **OPEN)] * 3, t)
    assert p.buttons == [(LEFT, True), (LEFT, False)]


def test_pinch_needs_to_open_past_release_threshold():
    p = FakePointer()
    c = Controller(p)
    between = (Config().pinch_press + Config().pinch_release) / 2
    feed(c, [Hand(0.5, 0.5, **PINCH), Hand(0.5, 0.5, between, 1.0), Hand(0.5, 0.5, 0.1, 1.0)])
    assert p.buttons == [(LEFT, True)]


def test_moving_while_pinched_drags():
    p = FakePointer()
    c = Controller(p)
    t = feed(c, [Hand(0.4, 0.5, **PINCH)] * 3)
    feed(c, [Hand(0.4 + 0.01 * i, 0.5, **PINCH) for i in range(20)], t)
    assert p.buttons == [(LEFT, True)]
    assert p.position[0] > 0.6


def test_middle_finger_pinch_is_right_click_and_buttons_do_not_overlap():
    p = FakePointer()
    c = Controller(p)
    feed(c, [Hand(0.5, 0.5, 1.0, 0.1), Hand(0.5, 0.5, 0.1, 0.1), Hand(0.5, 0.5, **OPEN)])
    assert p.buttons == [(RIGHT, True), (RIGHT, False)]


def test_losing_the_hand_releases_the_button_after_a_grace_period():
    p = FakePointer()
    c = Controller(p, Config(lost_grace=0.25))
    c.update(Hand(0.5, 0.5, **PINCH), 0.0)
    c.update(None, 0.1)
    assert p.buttons == [(LEFT, True)]
    c.update(None, 0.4)
    assert p.buttons == [(LEFT, True), (LEFT, False)]


def test_reappearing_hand_does_not_sweep_the_cursor_across_the_screen():
    p = FakePointer()
    c = Controller(p)
    t = feed(c, [Hand(0.3, 0.3, **OPEN)] * 5)
    c.update(None, t + 1.0)
    c.update(Hand(0.7, 0.7, **OPEN), t + 1.1)
    assert p.position == pytest.approx((0.9, 0.9))


def test_uinput_structs_match_kernel_layout():
    if struct.calcsize("l") != 8:
        pytest.skip("sizes below are for 64-bit Linux")
    assert struct.calcsize(pointer._USER_DEV) == 1116  # sizeof(struct uinput_user_dev)
    assert struct.calcsize(pointer._EVENT) == 24  # sizeof(struct input_event)


def test_auto_backend_prefers_uinput_on_wayland(monkeypatch):
    opened = []
    monkeypatch.setattr(pointer, "UinputPointer", lambda: opened.append("uinput"))
    monkeypatch.setattr(pointer, "X11Pointer", lambda: opened.append("x11"))
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    pointer.open_pointer()
    monkeypatch.delenv("WAYLAND_DISPLAY")
    pointer.open_pointer()
    assert opened == ["uinput", "x11"]

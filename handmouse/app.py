"""Command line entry point and the camera loop."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

from .gestures import Config, Controller, observe
from .pointer import PointerError, open_pointer

WINDOW = "handmouse"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = Config()
    p = argparse.ArgumentParser(prog="handmouse", description="Move and click the mouse pointer with your hand.")
    p.add_argument("--camera", default="0", help="camera index or device path (default: 0)")
    p.add_argument(
        "--backend",
        choices=("auto", "x11", "uinput"),
        default="auto",
        help="how to drive the pointer (default: x11 on X11 sessions, uinput on Wayland)",
    )
    p.add_argument("--no-preview", action="store_true", help="do not show the camera window")
    p.add_argument(
        "--margin",
        type=float,
        default=defaults.margin,
        help="fraction of the camera frame, per side, outside the active area; "
        f"larger means less hand travel (default: {defaults.margin})",
    )
    p.add_argument(
        "--pinch",
        type=float,
        default=defaults.pinch_press,
        help=f"how close thumb and fingertip must get to click; raise if clicks are missed (default: {defaults.pinch_press})",
    )
    p.add_argument(
        "--smoothing",
        type=float,
        default=defaults.smoothing,
        help=f"cursor steadiness; higher is steadier but laggier (default: {defaults.smoothing})",
    )
    p.add_argument("--model", type=Path, help="path to a hand_landmarker.task file (default: download and cache)")
    args = p.parse_args(argv)
    if not 0 <= args.margin < 0.5:
        p.error("--margin must be at least 0 and below 0.5")
    if args.pinch <= 0 or args.smoothing <= 0:
        p.error("--pinch and --smoothing must be positive")
    return args


def open_camera(source: str):
    import cv2

    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_V4L2)
        # MJPG is what lets most webcams deliver 30 fps; a one-frame buffer keeps latency down.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {source!r}. List cameras with `ls /dev/video*` and pick one with --camera.")
    return cap


def draw_preview(frame, landmarks, hand, controller: Controller, paused: bool) -> None:
    import cv2

    from .tracker import CONNECTIONS

    h, w = frame.shape[:2]
    m = controller.config.margin
    cv2.rectangle(frame, (int(m * w), int(m * h)), (int((1 - m) * w), int((1 - m) * h)), (255, 200, 0), 1)
    if landmarks:
        colour = (0, 0, 255) if controller.held else (0, 220, 0)
        px = [(int(x * w), int(y * h)) for x, y in landmarks]
        for a, b in CONNECTIONS:
            cv2.line(frame, px[a], px[b], colour, 2)
        cv2.circle(frame, (int(hand.x * w), int(hand.y * h)), 7, (255, 200, 0), -1)
    if paused:
        status = "PAUSED - press p to resume"
    elif hand is None:
        status = "no hand"
    else:
        status = f"pinch {min(hand.pinch_index, hand.pinch_middle):.2f}  {controller.held or ''}"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WINDOW, frame)


def run(args: argparse.Namespace) -> int:
    pointer = open_pointer(args.backend)

    # Imported late: these take a moment to load and are not needed to report a pointer error.
    import cv2

    from .tracker import HandTracker, ensure_model

    config = Config(margin=args.margin, pinch_press=args.pinch, pinch_release=args.pinch * 1.5, smoothing=args.smoothing)
    controller = Controller(pointer, config)
    tracker = HandTracker(ensure_model(args.model))
    cap = open_camera(args.camera)
    paused = False
    if not args.no_preview:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    print("handmouse is running. Pinch thumb and index finger to click; Ctrl+C to quit.", file=sys.stderr)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("The camera stopped delivering frames.", file=sys.stderr)
                return 1
            frame = cv2.flip(frame, 1)  # mirror, so moving right moves the cursor right
            now = time.monotonic()
            landmarks = tracker.detect(frame, now)
            hand = observe(landmarks, frame.shape[1] / frame.shape[0]) if landmarks else None
            if not paused:
                controller.update(hand, now)
            if not args.no_preview:
                draw_preview(frame, landmarks, hand, controller, paused)
                key = cv2.waitKey(1) & 0xFF
                closed = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
                if closed or key in (ord("q"), 27):
                    return 0
                if key == ord("p"):
                    paused = not paused
                    controller.reset()
    except KeyboardInterrupt:
        return 0
    finally:
        controller.reset()  # never leave a button held down
        cap.release()
        tracker.close()
        pointer.close()
        if not args.no_preview:
            cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Take the same clean exit path on SIGTERM as on Ctrl+C.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        return run(args)
    except (PointerError, RuntimeError) as e:
        print(f"handmouse: {e}", file=sys.stderr)
        return 1

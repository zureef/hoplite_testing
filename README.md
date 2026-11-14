# handmouse

Move and click the mouse pointer with your hand, tracked through a webcam. Linux, X11 and Wayland.

## Gestures

Hold your hand up with the palm facing the camera.

| Gesture | Action |
| --- | --- |
| Move your hand | Move the cursor |
| Pinch thumb and index finger, then let go | Left click (twice for a double click) |
| Pinch thumb and index finger, hold, and move | Drag |
| Pinch thumb and middle finger | Right click |

The cursor follows your palm, not a fingertip, so pinching does not nudge it. The blue rectangle in the
preview window is the area that maps to the whole screen.

## Install

Needs Python 3.10+ and a webcam.

```sh
python3 -m venv .venv
.venv/bin/pip install .
```

MediaPipe needs the system's EGL and OpenGL ES libraries. Desktop installs normally have them; if startup
fails with `libEGL.so.1: cannot open shared object file`, install them (`sudo apt install libegl1 libgles2`
on Debian/Ubuntu).

## Run

```sh
.venv/bin/handmouse
```

The first run downloads the hand tracking model (about 8 MB) to `~/.cache/handmouse`. In the preview window
press `p` to pause or resume control and `q` to quit; Ctrl+C in the terminal also quits. Your normal mouse
keeps working throughout.

| Option | Meaning |
| --- | --- |
| `--camera N` | Use camera `N` (or a device path) instead of camera 0 |
| `--no-preview` | Run without the camera window |
| `--margin F` | Shrink (larger `F`) or grow the active area; default 0.25 |
| `--pinch F` | Pinch distance that counts as a click; raise it if clicks are missed, lower it if you get accidental ones; default 0.22. The preview shows the live value |
| `--smoothing F` | Higher is a steadier but laggier cursor; default 1.0 |
| `--backend x11\|uinput` | Force a pointer backend (see below) |

## Wayland setup

On X11 sessions handmouse works out of the box. Wayland does not let applications move the pointer, so there
handmouse creates a virtual input device through `/dev/uinput`, which needs a one-time permission rule:

```sh
echo 'KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"' \
  | sudo tee /etc/udev/rules.d/60-handmouse.rules
sudo modprobe uinput
sudo udevadm control --reload-rules && sudo udevadm trigger /dev/uinput
```

This grants access to the user logged in at the machine. Check which session you have with
`echo $XDG_SESSION_TYPE`.

## Development

```sh
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
```

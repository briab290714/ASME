# Gesture Recognition Engine & Virtual Mouse

## Files

| File | Purpose |
|---|---|
| `hand_tracking_utils.py` | Shared hand-tracking + gesture-classification logic used by both Q1 and Q2. |
| `gesture_recognition.py` | **Q1** — real-time gesture recognition engine. |
| `virtual_mouse.py` | **Q2** — hand-controlled virtual mouse. |
| `mouse_config.json` | Sensitivity/smoothing settings for the virtual mouse (edit without touching code). |
| `test_gesture_logic.py` | Unit tests for the gesture classifier, using synthetic hand poses. |
| `test_mouse_logic.py` | Unit tests for the click/drag/scroll state machine, using a fake mouse backend. |
| `requirements.txt` | Python dependencies. |

## Setup

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
# Q1 — gesture recognition engine
python gesture_recognition.py

# Q2 — virtual mouse
python virtual_mouse.py
```

Press **q** in the video window to quit either program.

### Q1 keys
| Key | Action |
|---|---|
| `q` | Quit |
| `r` | Reset the gesture-password progress |
| `h` | Toggle the on-screen gesture legend |

### Q2 keys
| Key | Action |
|---|---|
| `q` | Quit |
| `c` | **Calibrate** your active movement zone (see below) |

### Calibrating the virtual mouse to your setup
Different desks, chairs, and camera placements mean "comfortable hand
movement range" varies a lot from person to person. Press **c**, then
pinch (thumb + index touching) at one corner of whatever range feels
natural, then pinch again at the opposite corner. The rectangle between
those two points becomes your `active_zone` — mapped to the *full*
screen — and is written straight into `mouse_config.json`, so it's
remembered automatically the next time you run the program. A green
rectangle is drawn on the video feed at all times so you can see exactly
which part of the frame currently maps to the screen; it updates the
moment calibration finishes.

If you never calibrate, `frame_margin` in the config is used as a
symmetric fallback instead.

## Running the tests (no camera needed)

```bash
python test_gesture_logic.py
python test_mouse_logic.py
```

Both test files build synthetic input (fake hand landmarks / a fake mouse
backend) so the core decision logic — angle-based finger classification,
the click/drag/scroll state machine, and the calibration state machine —
can be checked in isolation, independent of whatever gestures you happen
to perform in front of the camera during a live run. It's a good sanity
check to run once after cloning, and again any time you change a
threshold in the code.

---

## Q1 — Gesture Recognition Engine

### Gestures recognised
- **Open Palm** — all five fingers extended
- **Closed Fist** — all five fingers curled
- **Thumbs Up** — only the thumb extended, pointing upward
- **Victory** ✌️ — index + middle extended, ring + pinky curled
- **Pointing** — only the index finger extended
- **Pinch** — thumb tip touching index tip
- **Swipe Left / Swipe Right** *(dynamic)* — the wrist moving quickly and
  mostly horizontally over a short time window

### How classification works
Each finger's "extended or curled" state is decided using the **angle at
its middle joint** (`hand_tracking_utils.angle_at_joint`), not just a raw
y-coordinate comparison. A straight finger has a joint angle close to
180°; a curled one has a much smaller angle. This makes detection work
regardless of how the hand is rotated or tilted, because it is based on
joint geometry rather than absolute screen position.

The five finger states (thumb, index, middle, ring, pinky — each
`extended: bool`) are combined into gesture rules (e.g. "Victory" =
index & middle extended, ring & pinky curled). A **confidence score**
(0–1) is derived from how far past the threshold each angle is, and is
displayed alongside the gesture name.

**No hand detected** is handled explicitly — the video feed shows a
clear "No hand detected" message instead of crashing or showing stale
data. The engine also supports **multiple hands** at once (each is
labelled Left/Right with its own gesture).

### Brownie-point features implemented
- **Dynamic gestures**: swipe left/right, detected from the wrist's
  motion history over a rolling time window.
- **Gesture sequence / password system**: showing
  `Open Palm → Closed Fist → Victory → Open Palm` (each held briefly)
  displays "UNLOCKED!". The target sequence is a plain Python list
  (`PASSWORD_SEQUENCE` in `gesture_recognition.py`) that can be edited to
  define a custom "password".
- **Confidence score**: shown next to every detected gesture.
- A **`GestureStabilizer`** debounces raw per-frame classifications
  (a gesture must hold for several consecutive frames before it's
  "confirmed"), so the on-screen label and the password sequence don't
  flicker on single noisy frames.

### Performance / usability extras
- **Threaded camera capture** (`ThreadedCamera`): frames are read on a
  background thread instead of blocking the main loop on `cap.read()`,
  which noticeably cuts hand-to-screen lag on a real machine.
- **Camera auto-fallback** (`open_camera`): tries index 0, then 1, then 2,
  so it still finds your webcam if it isn't index 0 (common on laptops
  with both a built-in and a USB camera).
- **`MODEL_COMPLEXITY`** constant at the top of the file: set to `0` for
  the lightest/fastest MediaPipe hand model if you're on modest hardware,
  or leave at `1` for the more accurate default.
- On-screen **legend/cheat-sheet** (toggle with `h`) listing every
  gesture and the current password target, handy while recording a demo.
- The whole capture/processing loop is wrapped in `try/finally` so the
  camera thread and OpenCV windows are always cleaned up, even if
  something throws mid-loop.

---

## Q2 — Virtual Mouse

### Gesture → action mapping
| Gesture | Action |
|---|---|
| Open Palm / Pointing | Move pointer (follows index fingertip) |
| Pinch (quick tap) | Left click |
| Pinch (hold or move while pinched) | Drag |
| Closed Fist | Right click |
| Victory, moved up/down | Scroll *(bonus)* |

### Pointer control
The index fingertip's pixel position is mapped from an **active zone**
inside the camera frame to the full screen. That zone can come from
either:
- an **interactive calibration** (press `c`, pinch at two opposite
  corners of whatever hand-movement range is comfortable — see the
  "Calibrating" section above), which is saved to `mouse_config.json`
  and reused automatically next run, or
- a simple symmetric **`frame_margin`** fallback if you haven't
  calibrated yet.

A green rectangle is drawn on the video feed showing exactly which part
of the camera frame is currently "live" for pointer control.

Frame-to-frame jitter is removed with **exponential moving-average
smoothing** (`PointerMapper.map_and_smooth`): each frame the pointer
moves only a fraction of the way toward its new target, controlled by
`smoothing_factor` in `mouse_config.json`.

### Click / drag logic
`ClickStateMachine` is a small state machine (`idle → armed → dragging`,
plus a `right_click_held` state for the fist) that decides what a pinch
or fist means:
- A pinch that is released **quickly and without much movement** → a
  single left click.
- A pinch that is **held past `drag_hold_seconds`, or moved past
  `drag_move_threshold_px`** → a drag (mouse button held down, released
  when the pinch ends).
- A fist triggers **one** right click on the transition into the fist
  (not repeatedly while held), and won't fire again until cooldown
  passes and the fist is released and re-formed.
- A **`click_cooldown_seconds`** cooldown prevents accidental repeated
  clicks.
- If the hand **briefly leaves the frame while dragging**, the drag is
  kept alive for `hand_lost_grace_seconds` before being safely released
  — so a short tracking glitch doesn't drop your drag, but a real "hand
  gone" case doesn't leave the mouse button stuck down forever either.

This logic is deliberately kept separate from `pyautogui` itself (via an
injected `backend` object) — see `test_mouse_logic.py`, which exercises
every one of the behaviors above using a fake backend that just records
what it was told to do.

### Crash-proofing against pyautogui's screen-corner failsafe
`pyautogui` aborts with an exception the instant the pointer touches a
literal screen corner (its own built-in safety feature, left **on** here
on purpose — it's your emergency stop if the pointer ever runs away).
`PyAutoGuiBackend` catches that specific exception around every call
instead of letting it crash the program, and the video feed shows a
short warning ("Pointer hit a screen corner — move hand away") so you
know what happened and can just continue.

### Performance extras
- **Threaded camera capture** (`ThreadedCamera`, shared with Q1): reads
  frames on a background thread so the pointer keeps up with your hand
  instead of lagging behind the main loop's MediaPipe/OpenCV work.
- **Camera auto-fallback** (`open_camera`): tries a couple of camera
  indices automatically.
- **`MODEL_COMPLEXITY`** constant at the top of the file, same idea as
  in Q1 — drop to `0` if the pointer feels laggy.
- The main loop is wrapped in `try/finally`: if a drag happens to be in
  progress when the program exits (Ctrl+C, an error, etc.) the mouse
  button is explicitly released so it can never get stuck "held down".

### Adjustable sensitivity (config file)
All of the thresholds above live in `mouse_config.json`:

```json
{
  "smoothing_factor": 6,
  "frame_margin": 0.15,
  "active_zone": null,
  "click_cooldown_seconds": 0.4,
  "drag_hold_seconds": 0.25,
  "drag_move_threshold_px": 25,
  "hand_lost_grace_seconds": 0.6,
  "scroll_enabled": true,
  "scroll_sensitivity": 1800
}
```

`active_zone` starts out `null` and gets filled in automatically the
first time you calibrate with `c` — you don't need to edit it by hand,
though you're welcome to (it's `[x_min, y_min, x_max, y_max]` as
fractions of the camera frame, e.g. `[0.2, 0.2, 0.8, 0.8]`).

Lower `smoothing_factor` for a snappier (but jitterier) pointer; raise
`frame_margin` if you want a smaller physical hand-movement range to
cover the whole screen (only used while `active_zone` is `null`).

## Known limitations
- Angle/gesture thresholds were tuned for a hand held roughly facing the
  camera; extreme angles or very poor lighting can reduce accuracy —
  MediaPipe's own detection/tracking confidence thresholds
  (`HandTracker(detection_confidence=..., tracking_confidence=...)`)
  can be raised or lowered to trade off robustness vs. responsiveness.
- `pyautogui` requires an active graphical display; it will not run over
  a headless SSH session.
- In the rare case where a drag needs to be released (mouse-up) at the
  exact moment the pointer is sitting in a failsafe corner, that release
  call is itself skipped by the failsafe check, which could in theory
  leave the button logically "held" until the pointer next moves away
  from the corner. Calibrating your active zone (`c`) away from the
  screen edges avoids this in practice.

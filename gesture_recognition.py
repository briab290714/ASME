"""
gesture_recognition.py
------------------------
Q1: Gesture Recognition Engine

Opens the webcam, tracks up to 2 hands in real time, and recognises:

  STATIC gestures (held pose, classified every frame):
      Open Palm, Closed Fist, Thumbs Up, Victory, Pointing, Pinch

  DYNAMIC gestures (movement over several frames):
      Swipe Left, Swipe Right

  BONUS - gesture sequence "password":
      Show Open Palm -> Closed Fist -> Victory -> Open Palm (each held
      briefly) to trigger an "Unlocked!" message. This demonstrates using
      TEMPORAL information, not just a single frame.

Runs the webcam feed on a background thread (`ThreadedCamera`) so the
video stays responsive, and tries a couple of camera indices in case
index 0 isn't the right device on your machine (`open_camera`).

Controls:
    q  - quit
    r  - reset the gesture-sequence password progress
    h  - toggle the on-screen gesture legend/cheat-sheet

Run:
    python gesture_recognition.py
"""

import time
from collections import deque

import cv2

from hand_tracking_utils import (
    HandTracker,
    classify_static_gesture,
    WRIST,
    open_camera,
    ThreadedCamera,
)

# ---------------------------------------------------------------------------
# CONFIG - tweak these to tune sensitivity without touching the logic below
# ---------------------------------------------------------------------------
MAX_HANDS = 2
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
# 0 = fastest/lightest hand model, 1 = default (more accurate, a bit
# slower). If the video feels laggy on your machine, drop this to 0.
MODEL_COMPLEXITY = 1

# A static gesture must be seen for this many consecutive frames before we
# treat it as "confirmed" (this is what makes the sequence/password feature
# and the on-screen label stable instead of flickering frame to frame).
CONFIRM_FRAMES = 6

# --- swipe detection ---
SWIPE_HISTORY_SECONDS = 0.6      # how far back we look for motion
SWIPE_MIN_DISTANCE = 0.28        # min horizontal travel (fraction of frame width)
SWIPE_MAX_VERTICAL_RATIO = 0.6   # swipe must be mostly horizontal, not diagonal
SWIPE_COOLDOWN_SECONDS = 1.0     # ignore new swipes for a bit after one fires

# --- gesture password sequence (users can redefine this list!) ---
PASSWORD_SEQUENCE = ["Open Palm", "Closed Fist", "Victory", "Open Palm"]


# ---------------------------------------------------------------------------
# Helper class: turns noisy per-frame gesture guesses into a STABLE label,
# and also watches the wrist's motion history to detect swipes.
# ---------------------------------------------------------------------------
class GestureStabilizer:
    def __init__(self):
        self.current_gesture = "Unknown"
        self.current_conf = 0.0
        self.pending_gesture = "Unknown"
        self.pending_count = 0

        # (timestamp, normalised_x, normalised_y) of the wrist, recent history
        self.wrist_history = deque()
        self.last_swipe_time = 0.0

    def update_static(self, gesture, confidence):
        """Debounces a single frame's static-gesture guess."""
        if gesture == self.pending_gesture:
            self.pending_count += 1
        else:
            self.pending_gesture = gesture
            self.pending_count = 1

        if self.pending_count >= CONFIRM_FRAMES:
            self.current_gesture = gesture
            self.current_conf = confidence

        return self.current_gesture, self.current_conf

    def update_motion(self, wrist_x, wrist_y):
        """Feeds the latest wrist position in and checks for a swipe.
        Returns 'Swipe Left', 'Swipe Right', or None."""
        now = time.time()
        self.wrist_history.append((now, wrist_x, wrist_y))

        # drop anything older than our detection window
        while self.wrist_history and now - self.wrist_history[0][0] > SWIPE_HISTORY_SECONDS:
            self.wrist_history.popleft()

        if now - self.last_swipe_time < SWIPE_COOLDOWN_SECONDS:
            return None
        if len(self.wrist_history) < 2:
            return None

        oldest_t, oldest_x, oldest_y = self.wrist_history[0]
        dx = wrist_x - oldest_x
        dy = wrist_y - oldest_y

        if abs(dx) < SWIPE_MIN_DISTANCE:
            return None
        if abs(dy) > abs(dx) * SWIPE_MAX_VERTICAL_RATIO:
            return None  # too much vertical movement, probably not a swipe

        self.last_swipe_time = now
        self.wrist_history.clear()
        return "Swipe Right" if dx > 0 else "Swipe Left"


class PasswordSequenceDetector:
    """Watches CONFIRMED static gestures and checks them against a target
    sequence, e.g. Open Palm -> Closed Fist -> Victory -> Open Palm."""

    def __init__(self, target_sequence):
        self.target = target_sequence
        self.progress = []
        self.last_added = None
        self.unlocked_until = 0.0

    def feed(self, confirmed_gesture):
        # Only append when the confirmed gesture CHANGES, so holding a pose
        # doesn't spam the sequence with duplicates.
        if confirmed_gesture == self.last_added or confirmed_gesture == "Unknown":
            return
        self.last_added = confirmed_gesture
        self.progress.append(confirmed_gesture)
        self.progress = self.progress[-len(self.target):]  # keep it bounded

        if self.progress == self.target:
            self.unlocked_until = time.time() + 2.5  # show "Unlocked!" for 2.5s
            self.progress = []
            self.last_added = None

    def is_unlocked_now(self):
        return time.time() < self.unlocked_until

    def reset(self):
        self.progress = []
        self.last_added = None


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
def draw_text_with_background(frame, text, org, scale=0.8, color=(255, 255, 255),
                               bg=(0, 0, 0), thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    cv2.rectangle(frame, (x - 4, y - th - 6), (x + tw + 4, y + baseline + 2), bg, -1)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_password_progress(frame, detector):
    text = "Password: " + " -> ".join(detector.progress) if detector.progress else "Password: (show a gesture)"
    draw_text_with_background(frame, text, (10, frame.shape[0] - 45), scale=0.55,
                               color=(200, 200, 0), bg=(30, 30, 30))
    if detector.is_unlocked_now():
        draw_text_with_background(frame, "UNLOCKED!", (10, frame.shape[0] - 15), scale=0.8,
                                   color=(0, 255, 0), bg=(30, 30, 30))


def draw_legend(frame):
    """Small on-screen cheat-sheet, toggled with 'h'."""
    lines = [
        "Gestures: Open Palm | Closed Fist | Thumbs Up | Victory",
        "          Pointing | Pinch | Swipe Left/Right (move hand fast)",
        f"Password: {' -> '.join(PASSWORD_SEQUENCE)}",
        "Keys: q = quit | r = reset password | h = hide this help",
    ]
    x, y = frame.shape[1] - 470, 60
    for line in lines:
        draw_text_with_background(frame, line, (x, y), scale=0.5,
                                   color=(255, 255, 255), bg=(20, 20, 20))
        y += 26


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if cap is None:
        print("Could not open a webcam on any of the tried camera indices. "
              "Check that it is connected and not already in use by another app.")
        return

    # Reading frames on a background thread keeps the video feed responsive
    # even while the main loop is busy running MediaPipe + gesture logic.
    camera = ThreadedCamera(cap).start()
    tracker = HandTracker(max_hands=MAX_HANDS, model_complexity=MODEL_COMPLEXITY)

    # One stabilizer per tracked hand slot (index 0, 1, ...)
    stabilizers = [GestureStabilizer() for _ in range(MAX_HANDS)]
    password_detector = PasswordSequenceDetector(PASSWORD_SEQUENCE)

    swipe_message = ""
    swipe_message_until = 0.0
    show_legend = True

    prev_time = time.time()

    print("Gesture Recognition Engine (press 'q' to quit, 'h' for help)")
    print("Keys: q = quit | r = reset password progress | h = toggle help overlay")

    window_name= "Gesture Recognition Engine"
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue  # camera thread hasn't produced a frame yet

            frame = cv2.flip(frame, 1)  # mirror view feels natural to the user
            hands_found = tracker.process(frame)
            tracker.draw(frame, hands_found)

            if not hands_found:
                # --- Requirement: handle "no hand detected" gracefully ---
                draw_text_with_background(frame, "No hand detected", (10, 40),
                                           scale=0.9, color=(0, 0, 255), bg=(30, 30, 30))
            else:
                for i, hand in enumerate(hands_found):
                    if i >= len(stabilizers):
                        break  # safety guard, shouldn't happen given MAX_HANDS

                    gesture, conf = classify_static_gesture(hand["landmarks_norm"])
                    stable_gesture, stable_conf = stabilizers[i].update_static(gesture, conf)

                    wrist_x, wrist_y = hand["landmarks_norm"][WRIST]
                    swipe = stabilizers[i].update_motion(wrist_x, wrist_y)
                    if swipe:
                        swipe_message = f"{hand['label']} hand: {swipe}"
                        swipe_message_until = time.time() + 1.2

                    # Only the first tracked hand drives the password sequence,
                    # to keep the demo simple and unambiguous.
                    if i == 0:
                        password_detector.feed(stable_gesture)

                    # --- Display the gesture near that hand's wrist ---
                    wx, wy = hand["landmarks"][WRIST]
                    label = f"{hand['label']}: {stable_gesture} ({stable_conf * 100:.0f}%)"
                    draw_text_with_background(frame, label, (max(10, wx - 60), max(30, wy + 40)),
                                               scale=0.65, color=(0, 255, 255), bg=(30, 30, 30))

            # --- swipe banner (fades out after ~1.2s) ---
            if swipe_message and time.time() < swipe_message_until:
                draw_text_with_background(frame, swipe_message, (10, 80), scale=0.8,
                                           color=(255, 150, 0), bg=(30, 30, 30))

            draw_password_progress(frame, password_detector)
            if show_legend:
                draw_legend(frame)

            # --- FPS counter, just for reference while tuning ---
            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            draw_text_with_background(frame, f"FPS: {fps:.0f}", (frame.shape[1] - 120, 30),
                                       scale=0.6, color=(255, 255, 255), bg=(30, 30, 30))

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                password_detector.reset()
            elif key == ord('h'):
                show_legend = not show_legend
    finally:
        # Runs even if something above raises, so the camera thread and
        # OpenCV windows never get left in a broken/locked state.
        tracker.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

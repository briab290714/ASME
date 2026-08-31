"""
virtual_mouse.py
------------------
Q2: Virtual Mouse (Optimized with 1080p, Fullscreen/Window Toggle & Relaxed Sensitivity)
"""

import json
import os
import time

import cv2

from hand_tracking_utils import (
    HandTracker,
    classify_static_gesture,
    INDEX_TIP,
    open_camera,
    ThreadedCamera,
)

# --- 1080p Resolution Setup ---
CAMERA_INDEX = 0
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
MODEL_COMPLEXITY = 1

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "mouse_config.json")

# --- Tuned Sensitivity Parameters 
DEFAULT_CONFIG = {
    "smoothing_factor": 3,          # Faster, snappier pointer movement
    "frame_margin": 0.05,           # Easily reach all 4 screen corners
    "active_zone": None,            # [x_min, y_min, x_max, y_max] in 0-1 frame fractions
    "click_cooldown_seconds": 0.25, # Allows faster consecutive clicking
    "drag_hold_seconds": 0.15,      # Quicker drag initiation
    "drag_move_threshold_px": 12,   # Subtle movements trigger drags smoothly
    "hand_lost_grace_seconds": 0.6,
    "scroll_enabled": True,
    "scroll_sensitivity": 400,      # Responsive, effortless scrolling
}


def load_config():
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                user_config = json.load(f)
            config.update(user_config)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: couldn't read {CONFIG_PATH} ({e}); using defaults.")
    return config


def save_config(config):
    try:
        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)
    except OSError as e:
        print(f"Warning: couldn't save {CONFIG_PATH} ({e}).")


class PyAutoGuiBackend:
    def __init__(self):
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        self._pyautogui = pyautogui
        self.screen_w, self.screen_h = pyautogui.size()
        self.failsafe_triggered = False

    def _safe_call(self, fn, *args, **kwargs):
        try:
            fn(*args, **kwargs)
            self.failsafe_triggered = False
        except self._pyautogui.FailSafeException:
            self.failsafe_triggered = True

    def move_to(self, x, y):
        self._safe_call(self._pyautogui.moveTo, x, y)

    def left_click(self):
        self._safe_call(self._pyautogui.click, button="left")

    def right_click(self):
        self._safe_call(self._pyautogui.click, button="right")

    def mouse_down(self):
        self._safe_call(self._pyautogui.mouseDown, button="left")

    def mouse_up(self):
        self._safe_call(self._pyautogui.mouseUp, button="left")

    def scroll(self, amount):
        self._safe_call(self._pyautogui.scroll, int(amount))


class ClickStateMachine:
    def __init__(self, backend, config, clock=time.time):
        self.backend = backend
        self.cfg = config
        self.clock = clock

        self.state = "idle"
        self.pinch_start_time = 0.0
        self.pinch_start_pos = (0, 0)
        self.last_click_time = 0.0
        self.last_seen_time = self.clock()
        self._scroll_prev_y = None

    def _cooldown_ok(self):
        return (self.clock() - self.last_click_time) >= self.cfg["click_cooldown_seconds"]

    def hand_visible(self, gesture, screen_x, screen_y):
        self.last_seen_time = self.clock()

        if gesture == "Pinch":
            self._handle_pinch(screen_x, screen_y)
        else:
            self._release_pinch_if_needed()

        if gesture == "Closed Fist":
            self._handle_fist()
        else:
            if self.state == "right_click_held":
                self.state = "idle"

        if gesture == "Victory":
            self._handle_scroll(screen_y)
        else:
            self._scroll_prev_y = None

    def hand_missing(self):
        if self.state == "dragging":
            if self.clock() - self.last_seen_time > self.cfg["hand_lost_grace_seconds"]:
                self.backend.mouse_up()
                self.state = "idle"

    def _handle_pinch(self, x, y):
        if self.state == "idle":
            self.state = "armed"
            self.pinch_start_time = self.clock()
            self.pinch_start_pos = (x, y)
            return

        if self.state == "armed":
            held_long = (self.clock() - self.pinch_start_time) >= self.cfg["drag_hold_seconds"]
            moved_far = _dist(self.pinch_start_pos, (x, y)) >= self.cfg["drag_move_threshold_px"]
            if held_long or moved_far:
                self.state = "dragging"
                self.backend.mouse_down()

    def _release_pinch_if_needed(self):
        if self.state == "armed":
            if self._cooldown_ok():
                self.backend.left_click()
                self.last_click_time = self.clock()
            self.state = "idle"
        elif self.state == "dragging":
            self.backend.mouse_up()
            self.state = "idle"

    def _handle_fist(self):
        if self.state in ("idle",) and self._cooldown_ok():
            self.backend.right_click()
            self.last_click_time = self.clock()
            self.state = "right_click_held"

    def _handle_scroll(self, y):
        if not self.cfg.get("scroll_enabled", True):
            return
        if self._scroll_prev_y is not None:
            dy = self._scroll_prev_y - y
            if abs(dy) > 1:
                self.backend.scroll(dy / max(self.cfg["scroll_sensitivity"], 1) * 100)
        self._scroll_prev_y = y


def _dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


class PointerMapper:
    def __init__(self, frame_w, frame_h, screen_w, screen_h, config):
        self.frame_w, self.frame_h = frame_w, frame_h
        self.screen_w, self.screen_h = screen_w, screen_h
        self.smooth_x, self.smooth_y = screen_w / 2, screen_h / 2
        self._initialized = False
        self.reconfigure(config)

    def reconfigure(self, config):
        self.smoothing = max(config["smoothing_factor"], 1)

        zone = config.get("active_zone")
        if zone:
            x_min_f, y_min_f, x_max_f, y_max_f = zone
        else:
            margin = config["frame_margin"]
            x_min_f, x_max_f = margin, 1 - margin
            y_min_f, y_max_f = margin, 1 - margin

        self.x_min = self.frame_w * x_min_f
        self.x_max = self.frame_w * x_max_f
        self.y_min = self.frame_h * y_min_f
        self.y_max = self.frame_h * y_max_f

    def map_and_smooth(self, px, py):
        clamped_x = min(max(px, self.x_min), self.x_max)
        clamped_y = min(max(py, self.y_min), self.y_max)

        norm_x = (clamped_x - self.x_min) / max(self.x_max - self.x_min, 1e-6)
        norm_y = (clamped_y - self.y_min) / max(self.y_max - self.y_min, 1e-6)

        target_x = norm_x * self.screen_w
        target_y = norm_y * self.screen_h

        if not self._initialized:
            self.smooth_x, self.smooth_y = target_x, target_y
            self._initialized = True
        else:
            self.smooth_x += (target_x - self.smooth_x) / self.smoothing
            self.smooth_y += (target_y - self.smooth_y) / self.smoothing

        return self.smooth_x, self.smooth_y

    def zone_corners_px(self):
        return int(self.x_min), int(self.y_min), int(self.x_max), int(self.y_max)


class Calibrator:
    MIN_ZONE_SIZE = 0.08

    def __init__(self):
        self.active = False
        self.points = []
        self._was_pinching = False

    def start(self):
        self.active = True
        self.points = []
        self._was_pinching = False

    def cancel(self):
        self.active = False
        self.points = []

    def feed(self, gesture, index_tip_norm):
        if not self.active:
            return None

        is_pinching = gesture == "Pinch"
        if is_pinching and not self._was_pinching:
            self.points.append(index_tip_norm)
        self._was_pinching = is_pinching

        if len(self.points) < 2:
            return None

        (x1, y1), (x2, y2) = self.points
        x_min, x_max = sorted((x1, x2))
        y_min, y_max = sorted((y1, y2))
        self.active = False
        self.points = []

        if (x_max - x_min) < self.MIN_ZONE_SIZE or (y_max - y_min) < self.MIN_ZONE_SIZE:
            return None

        return [x_min, y_min, x_max, y_max]

    def status_text(self):
        if len(self.points) == 0:
            return "Calibrating: pinch at ONE corner of your comfortable zone"
        return "Calibrating: now pinch at the OPPOSITE corner"


def main():
    config = load_config()

    cap = open_camera(CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    if cap is None:
        print("Could not open webcam.")
        return

    ok, sample_frame = cap.read()
    if not ok:
        print("Could not read frame.")
        cap.release()
        return
    frame_h, frame_w = sample_frame.shape[:2]

    camera = ThreadedCamera(cap).start()
    backend = PyAutoGuiBackend()
    mapper = PointerMapper(frame_w, frame_h, backend.screen_w, backend.screen_h, config)
    click_sm = ClickStateMachine(backend, config)
    tracker = HandTracker(max_hands=1, model_complexity=MODEL_COMPLEXITY)
    calibrator = Calibrator()

    # --- Setup Window & Fullscreen Toggle Capability ---
    window_name = "Virtual Mouse (q=quit | f=fullscreen toggle | c=calibrate)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    is_fullscreen = True

    print("Virtual Mouse running.")
    print("Keys: q = quit | f = toggle full-screen/windowed | c = calibrate active zone")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                continue
            frame = cv2.flip(frame, 1)

            hands_found = tracker.process(frame)
            tracker.draw(frame, hands_found)

            if hands_found:
                hand = hands_found[0]
                gesture, conf = classify_static_gesture(hand["landmarks_norm"])
                index_px = hand["landmarks"][INDEX_TIP]

                if calibrator.active:
                    new_zone = calibrator.feed(gesture, hand["landmarks_norm"][INDEX_TIP])
                    if new_zone is not None:
                        config["active_zone"] = new_zone
                        save_config(config)
                        mapper.reconfigure(config)
                        print(f"Calibrated active zone: {new_zone}")
                else:
                    screen_x, screen_y = mapper.map_and_smooth(*index_px)

                    if gesture in ("Open Palm", "Pointing", "Pinch", "Victory"):
                        backend.move_to(screen_x, screen_y)

                    click_sm.hand_visible(gesture, screen_x, screen_y)

                cv2.putText(frame, f"{gesture} ({conf * 100:.0f}%) | state={click_sm.state}",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            else:
                if not calibrator.active:
                    click_sm.hand_missing()
                cv2.putText(frame, "No hand detected", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

            x1, y1, x2, y2 = mapper.zone_corners_px()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

            if calibrator.active:
                cv2.putText(frame, calibrator.status_text(), (20, frame.shape[0] - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if backend.failsafe_triggered:
                cv2.putText(frame, "Pointer hit screen corner (failsafe) - move hand away",
                            (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                is_fullscreen = not is_fullscreen
                prop = cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, prop)
            elif key == ord('c'):
                calibrator.start()
    finally:
        if click_sm.state == "dragging":
            backend.mouse_up()
        tracker.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
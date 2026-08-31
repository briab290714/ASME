"""
hand_tracking_utils.py
-----------------------
Shared building blocks used by BOTH the gesture recognition engine (Q1)
and the virtual mouse (Q2).

This file is split into two clear layers:

1. GEOMETRY / CLASSIFICATION (pure functions, no camera, no MediaPipe)
   These take a simple list of 21 (x, y) points -- one per hand landmark --
   and work out which fingers are up, which gesture is being made, etc.
   Because they don't depend on a webcam, they are easy to test and reuse.

2. HandTracker (a thin wrapper around MediaPipe + OpenCV)
   This is the only part that talks to the camera and to MediaPipe.
   It hands back plain (x, y) landmark lists that layer 1 understands.

MediaPipe's 21 hand landmarks (index -> name) look like this:

        8   12  16  20
        |    |   |   |
        7   11  15  19
        |    |   |   |
    4   6   10  14  18
     \\   |    |   |   |
      3  5    9  13  17
       \\ |    |   |   |
        2 ---- 0 --------   (0 = WRIST)
        |
        1

Named constants for the indices we use are defined below so the rest of
the code reads like English instead of magic numbers.
"""

import math

# ---------------------------------------------------------------------------
# 1. LANDMARK INDEX CONSTANTS
# ---------------------------------------------------------------------------
# These match the official MediaPipe Hands landmark model exactly.
WRIST = 0

THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Each of the 4 "long" fingers has the same joint pattern, so we can loop
# over them instead of repeating code four times.
FINGERS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

# ---------------------------------------------------------------------------
# 2. TUNABLE THRESHOLDS
# ---------------------------------------------------------------------------
# A finger is considered STRAIGHT (extended) if the angle at its middle
# joint is large (close to 180 degrees = a straight line).
# It is considered CURLED (bent into the palm) if that angle is small.
# Using an ANGLE instead of just "is the tip above the knuckle" makes the
# detector work even when the hand is rotated, tilted, or sideways --
# which is why the brief specifically calls out "angles between joints".
FINGER_STRAIGHT_ANGLE = 140.0   # degrees, >= this => extended
FINGER_CURLED_ANGLE = 90.0      # degrees, <= this => curled

THUMB_STRAIGHT_ANGLE = 150.0
THUMB_CURLED_ANGLE = 100.0

# A "pinch" (thumb tip touching index tip) is measured as a fraction of the
# hand's own size (wrist-to-middle-MCP distance) rather than a fixed pixel
# value, so it works whether the hand is close to or far from the camera.
PINCH_DISTANCE_RATIO = 0.30


# ---------------------------------------------------------------------------
# 3. LOW-LEVEL GEOMETRY HELPERS
# ---------------------------------------------------------------------------
def distance(p1, p2):
    """Euclidean distance between two (x, y) points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_at_joint(a, b, c):
    """
    Returns the angle ABC (the angle at point b, in degrees), formed by
    the two segments b->a and b->c.

    A straight finger has this angle close to 180 degrees.
    A fully curled finger has this angle closer to 30-60 degrees.
    """
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)

    if mag1 * mag2 == 0:
        return 180.0  # degenerate case, treat as "straight" to avoid noise

    # Clamp to [-1, 1] to guard against tiny floating point errors that
    # would otherwise make acos() crash.
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_angle))


def _clamp01(value):
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# 4. FINGER STATE (which fingers are up, and how confidently)
# ---------------------------------------------------------------------------
def get_finger_states(landmarks):
    """
    landmarks: a list of 21 (x, y) tuples, in MediaPipe order.

    Returns a dict like:
        {
          "thumb":  (True,  0.92),
          "index":  (True,  0.85),
          "middle": (False, 0.70),
          "ring":   (False, 0.95),
          "pinky":  (False, 0.88),
        }
    where the first value is "is this finger extended?" and the second is
    a 0-1 confidence for that particular decision (how far past the
    threshold the angle is).
    """
    states = {}

    # --- the four long fingers (index, middle, ring, pinky) ---
    for name, (mcp, pip, tip) in FINGERS.items():
        ang = angle_at_joint(landmarks[mcp], landmarks[pip], landmarks[tip])
        extended = ang >= FINGER_STRAIGHT_ANGLE

        if extended:
            # how far past the "straight" threshold are we? (more = more confident)
            conf = _clamp01((ang - FINGER_STRAIGHT_ANGLE) / (180.0 - FINGER_STRAIGHT_ANGLE) + 0.5)
        else:
            # how far below the "curled" threshold are we?
            conf = _clamp01((FINGER_CURLED_ANGLE - ang) / (FINGER_CURLED_ANGLE - 30.0) + 0.5)

        states[name] = (extended, conf)

    # --- thumb (different joint layout, so handled on its own) ---
    thumb_angle = angle_at_joint(landmarks[THUMB_CMC], landmarks[THUMB_MCP], landmarks[THUMB_TIP])
    thumb_extended = thumb_angle >= THUMB_STRAIGHT_ANGLE
    if thumb_extended:
        conf = _clamp01((thumb_angle - THUMB_STRAIGHT_ANGLE) / (180.0 - THUMB_STRAIGHT_ANGLE) + 0.5)
    else:
        conf = _clamp01((THUMB_CURLED_ANGLE - thumb_angle) / (THUMB_CURLED_ANGLE - 20.0) + 0.5)
    states["thumb"] = (thumb_extended, conf)

    return states


def hand_scale(landmarks):
    """A rough measure of 'how big is this hand in the frame', used to make
    distance-based checks (like pinch) independent of camera distance."""
    return max(distance(landmarks[WRIST], landmarks[MIDDLE_MCP]), 1e-6)


# ---------------------------------------------------------------------------
# 5. GESTURE CLASSIFICATION (static, single-frame gestures)
# ---------------------------------------------------------------------------
def classify_static_gesture(landmarks):
    """
    Looks at ONE frame of landmarks and decides which static gesture (if
    any) is being shown. Returns (gesture_name, confidence 0-1).

    Recognised gestures:
        "Open Palm", "Closed Fist", "Thumbs Up", "Victory", "Pinch"
    Falls back to "Unknown" if nothing matches confidently.
    """
    f = get_finger_states(landmarks)
    thumb_ext, thumb_conf = f["thumb"]
    index_ext, index_conf = f["index"]
    middle_ext, middle_conf = f["middle"]
    ring_ext, ring_conf = f["ring"]
    pinky_ext, pinky_conf = f["pinky"]

    scale = hand_scale(landmarks)
    pinch_dist = distance(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / scale

    # --- Open Palm: all five fingers extended ---
    if thumb_ext and index_ext and middle_ext and ring_ext and pinky_ext:
        conf = (thumb_conf + index_conf + middle_conf + ring_conf + pinky_conf) / 5
        return "Open Palm", conf

# --- Closed Fist: ALL FIVE fingers (including thumb) must be curled ---
    if not thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        conf = (thumb_conf + index_conf + middle_conf + ring_conf + pinky_conf) / 5
        return "Closed Fist", conf

    # --- Pinch: thumb tip and index tip touching. Checked first because it
    #     can happen alongside other finger states and is a very deliberate
    #     gesture (used heavily in Q2 for clicking). ---
    if pinch_dist < PINCH_DISTANCE_RATIO and (middle_ext or ring_ext or pinky_ext):
        conf = _clamp01(1.0 - (pinch_dist / PINCH_DISTANCE_RATIO))
        return "Pinch", conf

    # --- Victory / Peace: index + middle extended, ring + pinky curled ---
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        conf = (index_conf + middle_conf + ring_conf + pinky_conf) / 4
        return "Victory", conf

    # --- Thumbs Up: only the thumb extended, and it points upward on screen
    #     (image y grows downward, so "up" means a smaller y than the wrist) ---
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        points_up = landmarks[THUMB_TIP][1] < landmarks[WRIST][1]
        if points_up:
            conf = (thumb_conf + index_conf + middle_conf + ring_conf + pinky_conf) / 5
            return "Thumbs Up", conf

    # --- Pointing: only the index finger extended (handy extra gesture,
    #     also used to steer the pointer in the virtual mouse) ---
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        conf = (index_conf + middle_conf + ring_conf + pinky_conf) / 4
        return "Pointing", conf

    return "Unknown", 0.0


# ---------------------------------------------------------------------------
# 6. HandTracker: the only part that touches the camera / MediaPipe
# ---------------------------------------------------------------------------
class HandTracker:
    """
    Thin wrapper around MediaPipe Hands.

    Usage:
        tracker = HandTracker(max_hands=1)
        hands_found = tracker.process(frame)   # frame = a BGR OpenCV image
        for hand in hands_found:
            hand["landmarks"]   -> list of 21 (x, y) points in PIXEL coords
            hand["landmarks_norm"] -> the same, but normalised 0-1 (handy
                                       for distance/angle math that should
                                       not depend on image resolution)
            hand["label"]       -> "Left" or "Right"
        tracker.draw(frame, hands_found)        # draws skeleton overlay
        tracker.close()                         # release resources
    """

    def __init__(self, max_hands=2, detection_confidence=0.7, tracking_confidence=0.6,
                 model_complexity=1):
        # Imported here (rather than at module load time) so that this
        # file can still be imported and unit-tested on a machine where
        # MediaPipe/OpenCV aren't installed -- only actually using the
        # camera requires them.
        import cv2
        import mediapipe as mp

        self._cv2 = cv2
        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            # 0 = fastest/lightest model, 1 = default, more accurate but
            # slower. Drop this to 0 in the config constants if your
            # machine can't keep up a smooth frame rate.
            model_complexity=model_complexity,
        )

    def process(self, frame_bgr):
        """Runs detection on one BGR frame. Returns a list of hand dicts
        (empty list if no hand was found)."""
        h, w = frame_bgr.shape[:2]
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        hands_found = []
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                norm_pts = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                px_pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                label = "Right"
                if results.multi_handedness and i < len(results.multi_handedness):
                    label = results.multi_handedness[i].classification[0].label

                hands_found.append({
                    "landmarks": px_pts,
                    "landmarks_norm": norm_pts,
                    "label": label,
                    "mp_landmarks": hand_landmarks,  # kept for drawing only
                })
        return hands_found

    def draw(self, frame_bgr, hands_found):
        """Draws the MediaPipe hand skeleton on top of the frame."""
        for hand in hands_found:
            self._mp_drawing.draw_landmarks(
                frame_bgr,
                hand["mp_landmarks"],
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

    def close(self):
        self._hands.close()


# ---------------------------------------------------------------------------
# 7. Camera helpers
# ---------------------------------------------------------------------------
def open_camera(preferred_index=0, width=640, height=480, extra_indices=(1, 2)):
    """
    Opens a webcam with MJPEG compression and DirectShow backend (on Windows)
    to enable high native resolutions (e.g. 1080p).
    """
    import cv2
    import sys

    for index in (preferred_index, *extra_indices):
        # Use CAP_DSHOW on Windows to bypass standard driver resolution limits
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)

        if cap.isOpened():
            # Force MJPEG pixel format so USB bandwidth allows 1080p stream
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

            if width:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera opened successfully at resolution: {int(actual_w)}x{int(actual_h)}")

            return cap
        cap.release()
    return None


class ThreadedCamera:
    """
    Reads frames from a webcam on a background thread.

    Why: a plain `cap.read()` blocks the main thread until the next frame
    is ready. On a real machine, folding that wait into the same loop that
    also runs MediaPipe + gesture logic + drawing adds up to visible lag
    between moving your hand and seeing the pointer/label react. Reading
    on its own thread means the main loop only ever grabs "whatever the
    latest frame is right now" instead of waiting in line for the camera.

    Usage:
        cam = ThreadedCamera(cap)
        cam.start()
        ok, frame = cam.read()
        ...
        cam.stop()
    """

    def __init__(self, cap):
        import threading
        self._cap = cap
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._update, daemon=True)
        self._frame = None
        self._ok = False
        self._running = False

    def start(self):
        self._running = True
        self._thread.start()
        return self

    def _update(self):
        while self._running:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok, self._frame = ok, frame

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return self._ok, self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()

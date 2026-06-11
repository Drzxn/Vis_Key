"""
============================================================
  AI Virtual Mouse — Stage 3: Pinch-to-Click Gesture Control
============================================================
  Description : Hand tracking with inner-frame padding (DPI)
                and pinch-to-click gesture recognition.

  Key features
  ─────────────
  • FRAME_PADDING defines an inner bounding box inside the
    camera frame. Only this region drives the cursor.
  • np.interp() handles clamped linear interpolation.
  • Low-pass (lerp) smoothing from Stage 2 is preserved.
  • Pinch-to-Click: Measures Euclidean distance between 
    Index Finger Tip (Landmark #8) and Thumb Tip (Landmark #4).
  • Trigger pyautogui.click() when distance < 30 pixels.
  • Cooldown variable 'last_click_time' (0.5s limit) prevents
    rapid double/triple clicking.
  • Visual indicator: Changes active zone box to green, turns
    tips green, and draws a click circle when pinched.
  • Press Q to quit safely.

  Dependencies: opencv-python, mediapipe (>=0.10), pyautogui,
                numpy
  Model file  : hand_landmarker.task  (same directory)
  Run         : python hand_tracker.py
============================================================
"""

import cv2
import time
import math
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks        import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ── PyAutoGUI global settings ────────────────────────────────────────────────
pyautogui.FAILSAFE = False   # no FailSafeException at screen corners
pyautogui.PAUSE    = 0       # remove 0.1 s inter-call delay

# ── Landmark indices ─────────────────────────────────────────────────────────
INDEX_TIP = 8    # Index Finger Tip  — drives the cursor
THUMB_TIP = 4    # Thumb Tip         — pinch-to-click partner

# ── Hand skeleton connections (21-point MediaPipe topology) ──────────────────
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),           # thumb
    (0,5),(5,6),(6,7),(7,8),           # index finger
    (9,10),(10,11),(11,12),            # middle finger
    (13,14),(14,15),(15,16),           # ring finger
    (0,17),(17,18),(18,19),(19,20),    # pinky
    (5,9),(9,13),(13,17),              # palm knuckles
]

# ── Camera / capture settings ─────────────────────────────────────────────────
MODEL_PATH           = "hand_landmarker.task"
CAMERA_INDEX         = 0
FRAME_WIDTH          = 640
FRAME_HEIGHT         = 480
MAX_HANDS            = 1
DETECTION_CONFIDENCE = 0.60
TRACKING_CONFIDENCE  = 0.50
PRESENCE_CONFIDENCE  = 0.50

# ── DPI / Padding control ─────────────────────────────────────────────────────
FRAME_PADDING = 100

# Derived active-zone pixel boundaries (computed once at module level)
PAD_X_MIN = FRAME_PADDING
PAD_X_MAX = FRAME_WIDTH  - FRAME_PADDING   # 640 - 100 = 540
PAD_Y_MIN = FRAME_PADDING
PAD_Y_MAX = FRAME_HEIGHT - FRAME_PADDING   # 480 - 100 = 380

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTHING = 5

# ── Drawing colours (BGR) ─────────────────────────────────────────────────────
BONE_COLOR    = (0,   220, 100)   # green  — skeleton connections
JOINT_COLOR   = (255, 255, 255)   # white  — joint dots
INDEX_COLOR   = (255, 100,  20)   # orange — index tip marker
THUMB_COLOR   = (0,   200, 255)   # cyan   — thumb tip marker
ZONE_COLOR    = (0,   180, 255)   # amber  — active zone border
CURSOR_COLOR  = (0,   255, 180)   # mint   — HUD cursor readout


# ════════════════════════════════════════════════════════════════════════════
#  CLASS: HandTracker
# ════════════════════════════════════════════════════════════════════════════
class HandTracker:
    """Wraps MediaPipe Tasks HandLandmarker in VIDEO mode."""

    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode                  = RunningMode.VIDEO,
            num_hands                     = MAX_HANDS,
            min_hand_detection_confidence = DETECTION_CONFIDENCE,
            min_hand_presence_confidence  = PRESENCE_CONFIDENCE,
            min_tracking_confidence       = TRACKING_CONFIDENCE,
        )
        self._detector    = HandLandmarker.create_from_options(options)
        self._last_result = None

    def process(self, bgr_frame, timestamp_ms: int):
        """Run inference. timestamp_ms must be strictly increasing."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._last_result = self._detector.detect_for_video(img, timestamp_ms)
        return self._last_result

    def draw_skeleton(self, frame):
        """Draw hand bones and joint dots using pure OpenCV."""
        if not self.has_hand():
            return frame
        h, w = frame.shape[:2]
        for hand in self._last_result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], BONE_COLOR, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, pt, 4, JOINT_COLOR, cv2.FILLED)
        return frame

    def get_landmark(self, landmark_id: int):
        """
        Return the raw normalised landmark for the first detected hand.
        Landmark .x and .y are in [0, 1] relative to frame dimensions.
        Returns None when no hand is present.
        """
        if not self.has_hand():
            return None
        return self._last_result.hand_landmarks[0][landmark_id]

    def has_hand(self) -> bool:
        return bool(self._last_result and self._last_result.hand_landmarks)

    def release(self):
        self._detector.close()


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════
def draw_tip(frame, coords, color, label=""):
    """Highlight a landmark tip with a filled circle and text label."""
    if coords:
        cv2.circle(frame, coords, 11, color,         cv2.FILLED)
        cv2.circle(frame, coords, 11, (255,255,255), 1)   # white ring
        if label:
            cv2.putText(frame, label,
                        (coords[0]+14, coords[1]-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)


def draw_active_zone(frame, is_pinched=False):
    """
    Draw the padded active tracking zone as a visible rectangle.
    The corners of this box correspond to the screen corners.
    Keeping the finger inside this box gives full-screen cursor coverage.
    """
    color = (0, 255, 0) if is_pinched else ZONE_COLOR

    # Outer dashed-style effect: draw a slightly larger dark rect first
    cv2.rectangle(frame,
                  (PAD_X_MIN - 2, PAD_Y_MIN - 2),
                  (PAD_X_MAX + 2, PAD_Y_MAX + 2),
                  (0, 0, 0), 2)
    # Main zone border
    cv2.rectangle(frame,
                  (PAD_X_MIN, PAD_Y_MIN),
                  (PAD_X_MAX, PAD_Y_MAX),
                  color, 2)
    # Corner labels for clarity
    label = "ACTIVE ZONE [CLICK]" if is_pinched else "ACTIVE ZONE"
    cv2.putText(frame, label,
                (PAD_X_MIN + 6, PAD_Y_MIN - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 230, 0), 2, cv2.LINE_AA)


def px(frame, lm):
    """Convert a normalised MediaPipe landmark to frame pixel coordinates."""
    if lm is None:
        return None
    h, w = frame.shape[:2]
    return int(lm.x * w), int(lm.y * h)


def map_to_screen(finger_x: float, finger_y: float,
                  screen_w: int,   screen_h: int) -> tuple[int, int]:
    """
    Map a finger's pixel position within the camera frame to screen coordinates
    using the padded active zone as the input range.
    """
    screen_x = int(np.interp(finger_x,
                              [PAD_X_MIN, PAD_X_MAX],
                              [screen_w,  0          ]))   # inverted for mirror

    screen_y = int(np.interp(finger_y,
                              [PAD_Y_MIN, PAD_Y_MAX],
                              [0,          screen_h  ]))

    return screen_x, screen_y


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    # ── Screen resolution ────────────────────────────────────────────────────
    screen_width, screen_height = pyautogui.size()
    print(f"[INFO] Screen size : {screen_width} x {screen_height}")
    print(f"[INFO] Active zone : X {PAD_X_MIN}–{PAD_X_MAX}  "
          f"Y {PAD_Y_MIN}–{PAD_Y_MAX}  (padding={FRAME_PADDING}px)")

    # ── Camera ───────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check CAMERA_INDEX.")
        return

    tracker  = HandTracker()
    start_ns = time.perf_counter_ns()
    prev_t   = 0.0

    # Cooldown tracking variable for click gesture
    last_click_time = 0.0

    # Previous smoothed cursor position — seeded to screen centre
    plocX: float = screen_width  / 2
    plocY: float = screen_height / 2

    print("=" * 54)
    print("  AI Virtual Mouse  |  Press Q to quit")
    print(f"  Smoothing: {SMOOTHING}   Padding: {FRAME_PADDING} px")
    print("=" * 54)

    # ── Frame loop ───────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Mirror display (selfie / natural orientation)
        frame = cv2.flip(frame, 1)

        # Monotonic timestamp required by MediaPipe VIDEO mode
        ts_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        # ── MediaPipe inference ──────────────────────────────────────────────
        tracker.process(frame, ts_ms)

        # ── Get landmarks ────────────────────────────────────────────────────
        index_lm = tracker.get_landmark(INDEX_TIP)
        thumb_lm = tracker.get_landmark(THUMB_TIP)

        # Frame pixel positions (used for distance calculation & annotations)
        index_px = px(frame, index_lm)
        thumb_px = px(frame, thumb_lm)

        # ── Pinch-to-Click detection ─────────────────────────────────────────
        is_pinched = False
        if index_px is not None and thumb_px is not None:
            # Euclidean distance in pixels using the math library
            distance = math.sqrt((index_px[0] - thumb_px[0])**2 + (index_px[1] - thumb_px[1])**2)
            
            if distance < 30:
                is_pinched = True
                curr_time = time.time()
                # Cooldown check: only allow click if at least 0.5s passed
                if curr_time - last_click_time >= 0.5:
                    pyautogui.click()
                    last_click_time = curr_time
                    print(f"[CLICK] Pinch detected! Distance: {distance:.1f} px. Triggered pyautogui.click()")

        # ── Draw active zone (turns green if clicked/pinched) ────────────────
        draw_active_zone(frame, is_pinched)

        # ── Draw hand skeleton ───────────────────────────────────────────────
        tracker.draw_skeleton(frame)

        # Tip markers (turn green if clicked/pinched)
        index_color = (0, 255, 0) if is_pinched else INDEX_COLOR
        thumb_color = (0, 255, 0) if is_pinched else THUMB_COLOR
        draw_tip(frame, index_px, index_color, label="Index #8")
        draw_tip(frame, thumb_px, thumb_color, label="Thumb  #4")

        # Draw visual indicator circle around finger tips on pinch
        if is_pinched and index_px is not None and thumb_px is not None:
            mid_x = (index_px[0] + thumb_px[0]) // 2
            mid_y = (index_px[1] + thumb_px[1]) // 2
            # Outer indicator ring
            cv2.circle(frame, (mid_x, mid_y), 30, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "CLICK", (mid_x - 20, mid_y - 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

        # ── Cursor movement with padding + low-pass smoothing ────────────────
        if index_lm is not None:
            # Convert normalised landmark to FRAME pixel position (pre-flip)
            finger_x_raw = index_lm.x * FRAME_WIDTH
            finger_y_raw = index_lm.y * FRAME_HEIGHT

            # Map padded zone → screen coordinates
            screen_x, screen_y = map_to_screen(
                finger_x_raw, finger_y_raw,
                screen_width, screen_height
            )

            # Low-pass filter (lerp):
            clocX = plocX + (screen_x - plocX) / SMOOTHING
            clocY = plocY + (screen_y - plocY) / SMOOTHING

            # Move OS cursor
            pyautogui.moveTo(int(clocX), int(clocY))

            # Console output
            print(f"Moving mouse to: {int(clocX)}  {int(clocY)}")

            # HUD overlay on preview window
            cv2.putText(frame,
                        f"Mouse -> ({int(clocX)}, {int(clocY)})",
                        (10, FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        CURSOR_COLOR, 2, cv2.LINE_AA)

            # Update previous position for next frame's lerp
            plocX, plocY = clocX, clocY

        # Landmark pixel readout in console
        if tracker.has_hand():
            print(f"  Index Tip (#8): {str(index_px):<18}Thumb Tip (#4): {thumb_px}")

        # ── FPS overlay ──────────────────────────────────────────────────────
        cur_t  = time.perf_counter()
        fps    = 1.0 / (cur_t - prev_t) if prev_t else 0.0
        prev_t = cur_t
        draw_fps(frame, fps)

        # ── Render ───────────────────────────────────────────────────────────
        cv2.imshow("AI Virtual Mouse", frame)

        # ── Quit on Q ────────────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # ── Cleanup ──────────────────────────────────────────────────────────────
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("\n[INFO] Session ended cleanly.")


if __name__ == "__main__":
    main()

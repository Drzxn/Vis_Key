"""
============================================================
  AI Virtual Mouse — Stage 3+: Multi-Finger Tap & Scroll
============================================================
  Description : Hand tracking with dynamic finger tapping 
                detection and two-finger scroll mode.

  Key features
  ─────────────
  • Movement Boundaries (DPI Box) and low-pass smoothing
    from previous versions are fully preserved.
  • All Pinch/Euclidean distance thumb math has been removed.
  • Helper math.atan2 joint-angle and landmark Y-comparison
    functions are defined for flexion/tapping detection.
  • Left Click: Index Tip (#8) tapped below Index PIP (#6)
    while Middle Finger is raised. Trigger pyautogui.leftClick().
  • Right Click: Middle Tip (#12) tapped below Middle PIP (#10)
    while Index Finger is raised. Trigger pyautogui.rightClick().
  • Scroll Mode: Index and Middle Fingers both raised. Maps
    their average Y-displacement to pyautogui.scroll().
    Moves/Cursor updates are disabled during scroll mode.
  • Visual Overlays: Custom indicators for Left Tap, Right Tap,
    and Scroll mode.
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
INDEX_TIP = 8    # Index Finger Tip
INDEX_PIP = 6    # Index PIP Joint (knuckle for tap detection)
INDEX_MCP = 5    # Index Knuckle Joint

MIDDLE_TIP = 12  # Middle Finger Tip
MIDDLE_PIP = 10  # Middle PIP Joint (knuckle for tap detection)
MIDDLE_MCP = 9   # Middle Knuckle Joint

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
FRAME_PADDING = 150

# Derived active-zone pixel boundaries (computed once at module level)
PAD_X_MIN = FRAME_PADDING
PAD_X_MAX = FRAME_WIDTH  - FRAME_PADDING   # 640 - 100 = 540
PAD_Y_MIN = FRAME_PADDING
PAD_Y_MAX = FRAME_HEIGHT - FRAME_PADDING   # 480 - 100 = 380

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTHING = 3

# ── Gesture detection constants ────────────────────────────────────────────────
TAP_THRESHOLD = 0.01         # normalised distance tip must go below knuckle (larger Y)
RAISE_THRESHOLD = 0.01       # normalised distance tip must be above knuckle (smaller Y)
SCROLL_SENSITIVITY = 5       # multiplier for scroll velocity
CURSOR_SENSITIVITY = 1.8     # multiplier to make cursor reach screen corners easily
SWIPE_VELOCITY_THRESHOLD = 2.0  # normalised X speed threshold for desktop swipe

# ── Modes ────────────────────────────────────────────────────────────────────
MOVE = "MOVE"
LEFT_CLICK = "LEFT CLICK"
RIGHT_CLICK = "RIGHT CLICK"
SCROLL = "SCROLL"
TASK_VIEW = "TASK VIEW"
DESKTOP_LEFT = "DESKTOP LEFT"
DESKTOP_RIGHT = "DESKTOP RIGHT"
IDLE = "IDLE"

# ── Drawing colours (BGR) ─────────────────────────────────────────────────────
BONE_COLOR        = (0,   220, 100)   # green  — skeleton connections
JOINT_COLOR       = (255, 255, 255)   # white  — joint dots
INDEX_COLOR       = (255, 100,  20)   # orange — index tip marker
MIDDLE_COLOR      = (240, 160,  20)   # indigo — middle tip marker
ZONE_COLOR        = (0,   180, 255)   # amber  — active zone border
CURSOR_COLOR      = (0,   255, 180)   # mint   — HUD cursor readout

# Gesture Colors (BGR)
LEFT_CLICK_COLOR  = (0,   255, 0)     # green
RIGHT_CLICK_COLOR = (255, 0,   0)     # blue
SCROLL_COLOR      = (255, 0,   255)   # magenta


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
#  MATH / GESTURE HELPERS
# ════════════════════════════════════════════════════════════════════════════
def get_joint_angle(tip, pip, mcp):
    """
    Calculate the joint flexion angle at the PIP joint using math.atan2.
    This can be used to mathematically detect flexion.
    """
    if tip is None or pip is None or mcp is None:
        return 180.0
    # Vectors pip -> tip and pip -> mcp
    v1_x, v1_y = tip.x - pip.x, tip.y - pip.y
    v2_x, v2_y = mcp.x - pip.x, mcp.y - pip.y
    
    ang1 = math.atan2(v1_y, v1_x)
    ang2 = math.atan2(v2_y, v2_x)
    
    angle = math.degrees(ang1 - ang2)
    angle = abs(angle)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def is_finger_tapped(tip_lm, pip_lm, threshold=TAP_THRESHOLD) -> bool:
    """
    Check if a finger is tapped down by seeing if the tip's Y position
    is lower (higher coordinate value) than the PIP joint by a threshold.
    """
    if tip_lm is None or pip_lm is None:
        return False
    return tip_lm.y > (pip_lm.y + threshold)


def is_finger_raised(tip_lm, pip_lm, threshold=RAISE_THRESHOLD) -> bool:
    """
    Check if a finger is raised by seeing if the tip's Y position
    is higher (lower coordinate value) than the PIP joint by a threshold.
    """
    if tip_lm is None or pip_lm is None:
        return False
    return tip_lm.y < (pip_lm.y - threshold)


# ════════════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
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


def draw_active_zone(frame, is_scrolling=False):
    """
    Draw the padded active tracking zone as a visible rectangle.
    The corners of this box correspond to the screen corners.
    """
    color = SCROLL_COLOR if is_scrolling else ZONE_COLOR

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
    label = "ACTIVE ZONE [SCROLL MODE]" if is_scrolling else "ACTIVE ZONE"
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
    center_x = (PAD_X_MIN + PAD_X_MAX) / 2
    center_y = (PAD_Y_MIN + PAD_Y_MAX) / 2

    # Scale the coordinates around the active zone center to increase sensitivity
    finger_x_sens = center_x + (finger_x - center_x) * CURSOR_SENSITIVITY
    finger_y_sens = center_y + (finger_y - center_y) * CURSOR_SENSITIVITY

    screen_x = int(np.interp(finger_x_sens,
                              [PAD_X_MIN, PAD_X_MAX],
                              [0,         screen_w   ]))

    screen_y = int(np.interp(finger_y_sens,
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

    # Cooldown tracking variable for click gestures
    last_click_time = 0.0

    # Variable to monitor scroll Y displacement
    prev_scroll_y = None

    # Swipe velocity tracking variables
    prev_wrist_x = None
    prev_wrist_time = None
    swipe_cooldown_until = 0.0
    desktop_swipe_action = None
    desktop_swipe_time = 0.0

    # Task View gesture tracking variables
    three_fingers_start_time = None
    task_view_triggered = False
    last_task_view_time = 0.0
    task_view_display_time = 0.0

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
        index_pip_lm = tracker.get_landmark(INDEX_PIP)
        middle_lm = tracker.get_landmark(MIDDLE_TIP)
        middle_pip_lm = tracker.get_landmark(MIDDLE_PIP)

        # Pixel positions for display annotations and scroll computations
        index_px = px(frame, index_lm)
        middle_px = px(frame, middle_pip_lm)  # Knuckle display point
        # Get actual middle tip pixel for rendering tip
        middle_tip_px = px(frame, middle_lm)

        # Initialize gesture active flags
        left_click_active = False
        right_click_active = False
        scroll_mode = False
        task_view_active = False

        # Track horizontal swipe velocity using mirrored wrist X coordinate
        wrist_lm = tracker.get_landmark(0)
        velocity_x = 0.0
        curr_time = time.time()
        if wrist_lm is not None:
            # Mirrored X coordinate to match physical hand direction (left/right)
            mirrored_wrist_x = 1.0 - wrist_lm.x
            if prev_wrist_x is not None and prev_wrist_time is not None:
                dt = curr_time - prev_wrist_time
                if dt > 0:
                    velocity_x = (mirrored_wrist_x - prev_wrist_x) / dt
            prev_wrist_x = mirrored_wrist_x
            prev_wrist_time = curr_time
        else:
            prev_wrist_x = None
            prev_wrist_time = None

        # ── Gesture recognition math & triggers ──────────────────────────────
        if (index_lm is not None and index_pip_lm is not None and 
            middle_lm is not None and middle_pip_lm is not None):
            
            # Use Y-comparison rules
            index_raised  = is_finger_raised(index_lm, index_pip_lm, RAISE_THRESHOLD)
            middle_raised = is_finger_raised(middle_lm, middle_pip_lm, RAISE_THRESHOLD)
            index_tapped  = is_finger_tapped(index_lm, index_pip_lm, TAP_THRESHOLD)
            middle_tapped = is_finger_tapped(middle_lm, middle_pip_lm, TAP_THRESHOLD)

            # Check for folded fingers to detect closed fist
            ring_lm = tracker.get_landmark(16)
            ring_pip_lm = tracker.get_landmark(14)
            pinky_lm = tracker.get_landmark(20)
            pinky_pip_lm = tracker.get_landmark(18)

            index_folded = index_lm.y > index_pip_lm.y
            middle_folded = middle_lm.y > middle_pip_lm.y
            ring_folded = ring_lm is not None and ring_pip_lm is not None and ring_lm.y > ring_pip_lm.y
            pinky_folded = pinky_lm is not None and pinky_pip_lm is not None and pinky_lm.y > pinky_pip_lm.y

            is_fist = index_folded and middle_folded and ring_folded and pinky_folded

            # Get Ring and Pinky statuses for Task View
            ring_raised  = is_finger_raised(ring_lm, ring_pip_lm, RAISE_THRESHOLD) if (ring_lm and ring_pip_lm) else False
            pinky_raised = is_finger_raised(pinky_lm, pinky_pip_lm, RAISE_THRESHOLD) if (pinky_lm and pinky_pip_lm) else False
            
            three_fingers_raised = index_raised and middle_raised and ring_raised and pinky_folded

            # ── PRIORITY SYSTEM ──
            
            # 1. Desktop Swipe
            if is_fist and (velocity_x > SWIPE_VELOCITY_THRESHOLD or velocity_x < -SWIPE_VELOCITY_THRESHOLD):
                if curr_time > swipe_cooldown_until:
                    if velocity_x > SWIPE_VELOCITY_THRESHOLD:
                        pyautogui.hotkey('ctrl', 'win', 'right')
                        desktop_swipe_action = "RIGHT"
                        desktop_swipe_time = curr_time
                        swipe_cooldown_until = curr_time + 1.0
                        print(f"[SWIPE] Desktop Next triggered! Velocity: {velocity_x:.2f}")
                    else:
                        pyautogui.hotkey('ctrl', 'win', 'left')
                        desktop_swipe_action = "LEFT"
                        desktop_swipe_time = curr_time
                        swipe_cooldown_until = curr_time + 1.0
                        print(f"[SWIPE] Desktop Previous triggered! Velocity: {velocity_x:.2f}")

            # 2. Task View (Held for at least 1 second)
            elif three_fingers_raised:
                if three_fingers_start_time is None:
                    three_fingers_start_time = curr_time
                elif curr_time - three_fingers_start_time >= 1.0 and not task_view_triggered:
                    if curr_time - last_task_view_time >= 1.5:
                        pyautogui.hotkey('win', 'tab')
                        task_view_triggered = True
                        last_task_view_time = curr_time
                        task_view_display_time = curr_time
                        print("[TASK VIEW] win+tab triggered")
                task_view_active = True

            # 3. Scroll Mode
            elif is_fist:
                scroll_mode = True

            # 4. Right Click
            elif middle_tapped and index_raised:
                right_click_active = True
                if curr_time - last_click_time >= 0.5:
                    pyautogui.rightClick()
                    last_click_time = curr_time
                    print("[CLICK] Right Click triggered!")

            # 5. Left Click
            elif index_tapped and middle_raised:
                left_click_active = True
                if curr_time - last_click_time >= 0.5:
                    pyautogui.leftClick()
                    last_click_time = curr_time
                    print("[CLICK] Left Click triggered!")

            # Reset three fingers timer if not raised in this frame
            if not three_fingers_raised:
                three_fingers_start_time = None
                task_view_triggered = False

        # Determine the current MODE
        if desktop_swipe_action is not None and curr_time - desktop_swipe_time < 1.0:
            MODE = f"DESKTOP {desktop_swipe_action}"
        elif curr_time - task_view_display_time < 1.0:
            MODE = "TASK VIEW"
        elif scroll_mode:
            MODE = SCROLL
        elif left_click_active:
            MODE = LEFT_CLICK
        elif right_click_active:
            MODE = RIGHT_CLICK
        elif index_lm is not None:
            MODE = MOVE
        else:
            MODE = IDLE

        # ── Handle scrolling ─────────────────────────────────────────────────
        if scroll_mode:
            if index_px is not None and middle_tip_px is not None:
                # Combined average pixel Y position
                avg_y = (index_px[1] + middle_tip_px[1]) / 2.0
                
                if prev_scroll_y is not None:
                    # Inverted: moving hand UP decreases average Y (so displacement is positive)
                    displacement = prev_scroll_y - avg_y
                    scroll_amount = int(displacement * SCROLL_SENSITIVITY)
                    if scroll_amount != 0:
                        pyautogui.scroll(scroll_amount)
                        print(f"[SCROLL] scrolling displacement: {displacement:.1f} px -> amount: {scroll_amount}")
                prev_scroll_y = avg_y
        else:
            prev_scroll_y = None

        # ── Draw active zone (changes color to Magenta in Scroll Mode) ──────
        draw_active_zone(frame, scroll_mode)

        # ── Draw hand skeleton ───────────────────────────────────────────────
        tracker.draw_skeleton(frame)

        # Dynamic Tip marker coloring
        if left_click_active:
            index_color = LEFT_CLICK_COLOR
            middle_color = MIDDLE_COLOR
        elif right_click_active:
            index_color = INDEX_COLOR
            middle_color = RIGHT_CLICK_COLOR
        elif scroll_mode:
            index_color = SCROLL_COLOR
            middle_color = SCROLL_COLOR
        else:
            index_color = INDEX_COLOR
            middle_color = MIDDLE_COLOR

        draw_tip(frame, index_px, index_color, label="Index #8")
        draw_tip(frame, middle_tip_px, middle_color, label="Middle #12")

        # ── Visual indicators / overlays ─────────────────────────────────────
        if left_click_active and index_px is not None:
            # Highlight index tip with green ring
            cv2.circle(frame, index_px, 24, LEFT_CLICK_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, "L-CLICK", (index_px[0] - 25, index_px[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LEFT_CLICK_COLOR, 2, cv2.LINE_AA)

        if right_click_active and middle_tip_px is not None:
            # Highlight middle tip with blue ring
            cv2.circle(frame, middle_tip_px, 24, RIGHT_CLICK_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, "R-CLICK", (middle_tip_px[0] - 25, middle_tip_px[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, RIGHT_CLICK_COLOR, 2, cv2.LINE_AA)

        if scroll_mode and index_px is not None and middle_tip_px is not None:
            # Connection line and scroll rings
            cv2.circle(frame, index_px, 20, SCROLL_COLOR, 2, cv2.LINE_AA)
            cv2.circle(frame, middle_tip_px, 20, SCROLL_COLOR, 2, cv2.LINE_AA)
            cv2.line(frame, index_px, middle_tip_px, SCROLL_COLOR, 2, cv2.LINE_AA)
            
            mid_x = (index_px[0] + middle_tip_px[0]) // 2
            mid_y = (index_px[1] + middle_tip_px[1]) // 2
            cv2.putText(frame, "SCROLL", (mid_x - 25, mid_y - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, SCROLL_COLOR, 2, cv2.LINE_AA)

        # ── On-screen temporary overlays for swipe/task view ─────────────────
        if desktop_swipe_action is not None and curr_time - desktop_swipe_time < 1.0:
            cv2.putText(frame, f"DESKTOP {desktop_swipe_action}",
                        (FRAME_WIDTH // 2 - 140, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
        elif curr_time - task_view_display_time < 1.0:
            cv2.putText(frame, "TASK VIEW",
                        (FRAME_WIDTH // 2 - 80, FRAME_HEIGHT // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)

        # ── Cursor movement (Only active when NOT in Scroll/Task View/Swipe Mode) ──
        if index_lm is not None and not scroll_mode and not task_view_active and desktop_swipe_action is None:
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

        elif scroll_mode:
            # HUD overlay indicating scroll mode
            cv2.putText(frame,
                        "SCROLL MODE ACTIVE",
                        (10, FRAME_HEIGHT - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        SCROLL_COLOR, 2, cv2.LINE_AA)

        # Landmark pixel readout in console
        if tracker.has_hand():
            print(f"  Index Tip (#8): {str(index_px):<18}Middle Tip (#12): {middle_tip_px}")

        # ── FPS overlay ──────────────────────────────────────────────────────
        cur_t  = time.perf_counter()
        fps    = 1.0 / (cur_t - prev_t) if prev_t else 0.0
        prev_t = cur_t
        draw_fps(frame, fps)

        # ── Mode overlay ─────────────────────────────────────────────────────
        cv2.putText(frame, f"MODE: {MODE}",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 255, 255), 2, cv2.LINE_AA)

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

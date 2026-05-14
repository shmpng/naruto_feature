"""
landmarks.py
All MediaPipe Face Landmarker indices and normalized eye-socket shapes
used across the Sage Mode + Iris filters.
"""

# ── Iris landmark clusters ────────────────────────────────────────────────────
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# ── EAR (Eye Aspect Ratio) key points ────────────────────────────────────────
L_TOP, L_BOT = 159, 145
R_TOP, R_BOT = 386, 374
L_L,   L_R   = 33,  133
R_L,   R_R   = 362, 263

# ── Eyelid contour rows (iris sticker clipping) ───────────────────────────────
L_TOP_ROW = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_BOT_ROW = [33,   7, 163, 144, 145, 153, 154, 155, 133]
R_TOP_ROW = [362, 398, 384, 385, 386, 387, 388, 466, 263]
R_BOT_ROW = [362, 382, 381, 380, 374, 373, 390, 249, 263]

# ── Eye corners (eyeshadow) ───────────────────────────────────────────────────
R_INNER, R_OUTER = 362, 263
L_INNER, L_OUTER = 133,  33

# ── Full eyelid outlines (eyeshadow clipping + lash lines) ───────────────────
R_UPPER_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362]
R_LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]
L_UPPER_LID = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_LOWER_LID = [33,   7, 163, 144, 145, 153, 154, 155, 133]

# ── Eyeshadow socket shapes ───────────────────────────────────────────────────
# Normalized to eye_width = 1.0 unit (captured at ~71 px eye width).
# RIGHT eye socket
_BASELINE = 71.0

_RAW_SOCKET_RIGHT = [
    [-28, 6], [-28, 2], [-28, -10], [-14, -25], [1, -31],
    [16, -30], [28, -25], [36, -21], [43, -16], [43, -16],
    [33, 0],  [23, 3],  [11, 5],   [-4, 7],   [-17, 7],
    [-28, 6], [-28, 6], [-28, 6],
]
NORM_SOCKET_RIGHT = [
    (dx / _BASELINE, dy / _BASELINE) for dx, dy in _RAW_SOCKET_RIGHT
]

# LEFT eye socket (mirrored)
_RAW_SOCKET_LEFT = [
    [28, 6],  [28, 2],  [28, -10], [14, -25], [-1, -31],
    [-16, -30], [-28, -25], [-36, -21], [-43, -16], [-43, -16],
    [-33, 0], [-23, 3], [-11, 5],  [4, 7],    [17, 7],
    [28, 6],  [28, 6],  [28, 6],
]
NORM_SOCKET_LEFT = [
    (dx / _BASELINE, dy / _BASELINE) for dx, dy in _RAW_SOCKET_LEFT
]

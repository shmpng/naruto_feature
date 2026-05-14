"""
Sage Mode — RIGHT EYE ONLY (DISTANCE-AWARE SCALING)
The shape scales with eye size — bigger when close, smaller when far.
FILES: realorange.jpg + face_landmarker.task in same folder
INSTALL: pip install mediapipe opencv-python numpy
"""

import cv2
import numpy as np
import os
import urllib.request

try:
    import mediapipe as mp
    print(f"✅ MediaPipe v{mp.__version__}")
except ImportError:
    print("❌ pip install mediapipe"); exit(1)

from mediapipe.tasks.python import vision as _v
from mediapipe.tasks.python.core import base_options as _b

R_L, R_R = 362, 263

# ================================================================
# PERMANENT SHAPE — normalized to eye width = 1.0 unit
# These were originally captured at ~71px eye width, so divide by 71
# ================================================================

RAW_SOCKET = [
    [-28, 6], [-28, 2], [-28, -10], [-14, -25], [1, -31],
    [16, -30], [28, -25], [36, -21], [43, -16], [43, -16],
    [33, 0], [23, 3], [11, 5], [-4, 7], [-17, 7],
    [-28, 6], [-28, 6], [-28, 6]
]

RAW_PUNCH = [
    [-32, 15], [-28, 5], [-21, -3], [-10, -7], [0, -8],
    [12, -8], [20, -7], [27, -6], [32, -5], [32, -5],
    [25, 0], [17, 7], [8, 12], [-3, 17], [-13, 17],
    [-23, 15], [-28, 15], [-32, 15]
]

# Baseline eye width the shape was captured at (pixels)
BASELINE_EYE_WIDTH = 71.0

# Normalize to [-1, 1] space relative to eye width
NORM_SOCKET = [(dx / BASELINE_EYE_WIDTH, dy / BASELINE_EYE_WIDTH) for dx, dy in RAW_SOCKET]
NORM_PUNCH  = [(dx / BASELINE_EYE_WIDTH, dy / BASELINE_EYE_WIDTH) for dx, dy in RAW_PUNCH]


def download_model(path):
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print(f"Downloading {path}…")
        urllib.request.urlretrieve(url, path)
    return path


def lm_px(lms, idx, W, H):
    return np.array([lms[idx].x * W, lms[idx].y * H], dtype=np.float32)


def apply_sage_eye(frame, lms, W, H, orange_img):
    left_pt  = lm_px(lms, R_L, W, H)
    right_pt = lm_px(lms, R_R, W, H)

    # Current eye width in pixels — this is our scale factor
    eye_width = np.linalg.norm(right_pt - left_pt)
    if eye_width < 4:
        return

    eye_cx = int((left_pt[0] + right_pt[0]) / 2)
    eye_cy = int((left_pt[1] + right_pt[1]) / 2)

    # Scale normalized coords by current eye width → proper pixel offsets
    socket_poly = np.array([
        [eye_cx + int(dx * eye_width), eye_cy + int(dy * eye_width)]
        for dx, dy in NORM_SOCKET
    ], dtype=np.int32)

    # ── REAL EYELID LANDMARKS ─────────────────────────────────────
    # Upper lid goes from outer corner → peak → inner corner (top arc)
    # Lower lid goes from outer corner → trough → inner corner (bottom arc)
    UPPER_LID = [263, 466, 388, 387, 386, 385, 384, 398, 362]
    LOWER_LID = [263, 249, 390, 373, 374, 380, 381, 382, 362]

    upper_lid_pts = np.array(
        [[int(lms[i].x * W), int(lms[i].y * H)] for i in UPPER_LID],
        dtype=np.int32)
    lower_lid_pts = np.array(
        [[int(lms[i].x * W), int(lms[i].y * H)] for i in LOWER_LID],
        dtype=np.int32)

    # Build the visible eye opening polygon:
    # upper lid arc + lower lid arc joined = the open eye shape
    # Orange is clipped to this — closes with the eye naturally
    eye_opening_poly = np.vstack([upper_lid_pts, lower_lid_pts[::-1]])

    bx1 = max(0, int(socket_poly[:, 0].min()) - 4)
    by1 = max(0, int(socket_poly[:, 1].min()) - 4)
    bx2 = min(W, int(socket_poly[:, 0].max()) + 4)
    by2 = min(H, int(socket_poly[:, 1].max()) + 4)
    box_w = bx2 - bx1
    box_h = by2 - by1
    if box_w < 4 or box_h < 4:
        return

    oh, ow = orange_img.shape[:2]
    scale    = box_w / ow
    scaled_w = box_w
    scaled_h = max(int(oh * scale), box_h)
    orange_s = cv2.resize(orange_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    eye_bottom = int(socket_poly[:, 1].max())
    oy_start = eye_bottom - scaled_h
    fy1 = max(0, oy_start)
    fy2 = min(H, eye_bottom)
    fx1 = bx1
    fx2 = min(W, bx1 + scaled_w)

    sy1 = max(0, fy1 - oy_start)
    sy2 = min(scaled_h, sy1 + (fy2 - fy1))
    sx1 = 0
    sx2 = min(scaled_w, fx2 - fx1)

    fy2 = fy1 + (sy2 - sy1)
    fx2 = fx1 + (sx2 - sx1)
    if fy2 <= fy1 or fx2 <= fx1:
        return

    orange_patch = orange_s[sy1:sy2, sx1:sx2, :3]

    # Socket mask = the orange skin area (around the eye)
    socket_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(socket_mask, [socket_poly], 255)

    # Eye opening mask = the actual open area between eyelids (dynamic)
    # Orange is NOT painted here — this is what clips with eye close
    eye_open_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(eye_open_mask, [eye_opening_poly], 255)

    # Paint only on socket but NOT inside the visible eye opening
    paint_mask = cv2.bitwise_and(socket_mask, cv2.bitwise_not(eye_open_mask))

    ry1 = max(0, min(by1, fy1))
    ry2 = min(H, max(by2, fy2))
    rx1 = max(0, bx1)
    rx2 = min(W, bx2)
    if ry2 <= ry1 or rx2 <= rx1:
        return

    roi      = frame[ry1:ry2, rx1:rx2].astype(np.float32)
    mask_roi = paint_mask[ry1:ry2, rx1:rx2].astype(np.float32) / 255.0

    canvas = np.zeros_like(roi)
    py1 = max(0, fy1 - ry1)
    py2 = min(ry2 - ry1, py1 + (fy2 - fy1))
    px1 = max(0, fx1 - rx1)
    px2 = min(rx2 - rx1, px1 + (fx2 - fx1))
    ph = py2 - py1
    pw = px2 - px1
    if ph > 0 and pw > 0:
        canvas[py1:py2, px1:px2] = orange_patch[:ph, :pw]

    alpha   = mask_roi[:, :, None] * 0.93
    blended = canvas * alpha + roi * (1.0 - alpha)
    frame[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)

    # ── BLACK LASH LINES — reuse real lid pts computed above ──────
    lash_thickness = max(1, int(eye_width * 0.035))
    cv2.polylines(frame, [upper_lid_pts], isClosed=False,
                  color=(0, 0, 0), thickness=lash_thickness,
                  lineType=cv2.LINE_AA)
    cv2.polylines(frame, [lower_lid_pts], isClosed=False,
                  color=(0, 0, 0), thickness=lash_thickness,
                  lineType=cv2.LINE_AA)
    # ─────────────────────────────────────────────────────────────

    # Debug: show current eye width
    cv2.putText(frame, f"eye_w: {eye_width:.0f}px", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)


def main():
    print("\n🦊 SAGE MODE — DISTANCE-AWARE SCALING\n")
    print("Shape grows/shrinks with your distance from camera.\n")

    det = _v.FaceLandmarker.create_from_options(
        _v.FaceLandmarkerOptions(
            base_options=_b.BaseOptions(
                model_asset_path=download_model("face_landmarker.task")),
            num_faces=1,
            min_face_detection_confidence=0.4,
            min_tracking_confidence=0.4))

    orange_path = "realorange.jpg"
    if not os.path.exists(orange_path):
        print(f"❌ '{orange_path}' not found in {os.getcwd()}")
        return
    orange = cv2.imread(orange_path)
    print(f"✅ Orange: {orange.shape[1]}×{orange.shape[0]}")

    cap = cv2.VideoCapture(0)
    for rw, rh in [(1280, 720), (854, 480), (640, 480)]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
        if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) >= rw - 10:
            break
    print(f"📷 {int(cap.get(3))}×{int(cap.get(4))}  |  Q=quit\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        H, W = frame.shape[:2]

        res = det.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if res.face_landmarks:
            lms = res.face_landmarks[0]
            apply_sage_eye(frame, lms, W, H, orange_img=orange)
            cv2.putText(frame, "SAGE MODE", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO FACE", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.putText(frame, "Q=quit", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imshow("Sage Mode — Distance Aware", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
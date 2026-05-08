"""
Eye Filter — Standalone
FILES NEEDED (same folder):
  eye_overlay.png      ← yellow minus circle
  eyeshadow.png        ← orange eyeshadow
  face_landmarker.task ← auto-downloaded on first run

INSTALL:  pip install mediapipe opencv-python numpy
KEYS:     Q = Quit   S = Reset filter
"""

import cv2
import numpy as np
import os
import urllib.request
import time

try:
    import mediapipe as mp
    print(f"✅ MediaPipe v{mp.__version__}")
except ImportError:
    print("❌ pip install mediapipe"); exit(1)

from mediapipe.tasks.python import vision as _v
from mediapipe.tasks.python.core import base_options as _b

# ── Landmark indices ─────────────────────────────────────────
# Iris: center + 4 edge points
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

# Eyelid top/bottom single points (for EAR)
L_TOP, L_BOT = 159, 145
R_TOP, R_BOT = 386, 374

# Eye corners (for EAR width)
L_L, L_R = 33,  133
R_L, R_R = 362, 263

# Dense eyelid edge rows for smooth clipping polygon
#   top row: upper eyelid curve (left→right)
L_TOP_ROW = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_BOT_ROW = [33, 7,   163, 144, 145, 153, 154, 155, 133]
R_TOP_ROW = [362,398, 384, 385, 386, 387, 388, 466, 263]
R_BOT_ROW = [362,382, 381, 380, 374, 373, 390, 249, 263]

# Eyeshadow brow anchors
L_BROW_OUT = 46
R_BROW_OUT = 276

# ── Tuning ───────────────────────────────────────────────────
EAR_OPEN     = 0.22
EAR_CLOSED   = 0.13
CLOSE_HOLD   = 3.0
ALPHA_IRIS   = 0.92
ALPHA_SHADOW = 0.72


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════
def download_model(path):
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print(f"   Downloading {path} …"); urllib.request.urlretrieve(url, path)
    return path


def remove_white_bg(img):
    """Remove white/near-white background → transparent."""
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    white = (b > 200) & (g > 200) & (r > 200)
    a = np.ones(img.shape[:2], np.uint8) * 255
    a[white] = 0
    return cv2.merge([b, g, r, a])

def remove_black_bg(img):
    """Remove black/near-black background → transparent."""
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    black = (b < 40) & (g < 40) & (r < 40)
    a = np.ones(img.shape[:2], np.uint8) * 255
    a[black] = 0
    return cv2.merge([b, g, r, a])

def load_overlay(path, name, bg="white"):
    """
    bg = 'white'  → remove white background  (eye_overlay)
    bg = 'black'  → remove black background  (eyeshadow)
    """
    if not os.path.exists(path):
        print(f"⚠️  {name} not found at '{path}' — skipping"); return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  Cannot read {path}"); return None
    # Always work in BGRA
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    # Strip background if no existing alpha info
    if bg == "white":
        img = remove_white_bg(img)
    elif bg == "black":
        img = remove_black_bg(img)
    print(f"✅ Loaded {name}: {img.shape[1]}×{img.shape[0]}")
    return img


def ear(lms, t, b, l, r):
    return abs(lms[t].y - lms[b].y) / (abs(lms[l].x - lms[r].x) + 1e-6)


def get_iris(lms, indices, w, h):
    """Returns (cx, cy, radius) of iris in pixel space."""
    cx = lms[indices[0]].x * w
    cy = lms[indices[0]].y * h
    # radius = mean distance from center to 4 edge points
    radii = [np.hypot(lms[i].x*w - cx, lms[i].y*h - cy) for i in indices[1:]]
    radius = max(int(np.mean(radii)), 4)
    return int(cx), int(cy), radius


# ════════════════════════════════════════════════════════════
#  Core: draw iris sticker with tight eyelid clipping
# ════════════════════════════════════════════════════════════
def draw_iris_sticker(frame, lms, iris_idx, top_row, bot_row, ov, ear_val):
    """
    Places overlay on iris, clipped by the real eyelid polygon.
    The clip tightens as the eye closes — giving a natural
    open/close animation that follows the actual eyelid shape.
    """
    if ov is None:
        return

    # openness 0.0 (fully closed) → 1.0 (fully open)
    openness = float(np.clip(
        (ear_val - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED), 0.0, 1.0))
    if openness < 0.02:
        return

    H, W = frame.shape[:2]
    cx, cy, radius = get_iris(lms, iris_idx, W, H)

    # Scale overlay to iris diameter × a small padding factor so it
    # covers the whole iris including limbus ring
    scale  = 1.15
    diam   = max(int(radius * 2 * scale), 8)
    half   = diam // 2

    # Resize overlay to iris size
    resized = cv2.resize(ov, (diam, diam), interpolation=cv2.INTER_AREA)

    # ── Build eyelid clipping polygon in sticker-local coords ──
    def lm_px(idx):
        return np.array([lms[idx].x * W, lms[idx].y * H])

    def to_local(px, py):
        return (int(round(px - (cx - half))),
                int(round(py - (cy - half))))

    # Upper eyelid curve (left → right)
    upper = [to_local(*lm_px(i)) for i in top_row]
    # Lower eyelid curve (right → left, so polygon closes)
    lower = [to_local(*lm_px(i)) for i in reversed(bot_row)]

    poly_pts = upper + lower
    poly_np  = np.array(poly_pts, dtype=np.int32)

    # Build mask from polygon
    lid_mask = np.zeros((diam, diam), np.uint8)
    cv2.fillPoly(lid_mask, [poly_np], 255)

    # ── Additional eyelash inset: erode mask slightly ──────────
    # This trims the very edge so the sticker doesn't bleed past
    # the lash line — mimics how lashes cut the visible iris edge
    inset_px = max(1, int(radius * 0.08))
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (inset_px*2+1, inset_px*2+1))
    lid_mask = cv2.erode(lid_mask, kernel, iterations=1)

    # ── Composite onto frame ────────────────────────────────────
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + diam,  y1 + diam

    # Frame bounds clamp
    fx1 = max(0, x1);  fy1 = max(0, y1)
    fx2 = min(W, x2);  fy2 = min(H, y2)
    if fx2 <= fx1 or fy2 <= fy1:
        return

    # Corresponding slice in the sticker
    ox1 = fx1 - x1;  oy1 = fy1 - y1
    ox2 = ox1 + (fx2 - fx1)
    oy2 = oy1 + (fy2 - fy1)

    roi      = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    ov_crop  = resized[oy1:oy2, ox1:ox2]
    mask_crop= lid_mask[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0

    # Overlay alpha channel
    if ov_crop.shape[2] == 4:
        ov_a = ov_crop[:, :, 3:].astype(np.float32) / 255.0
    else:
        ov_a = np.ones((*ov_crop.shape[:2], 1), np.float32)

    # Final blend weight = eyelid mask × overlay alpha × openness × max_alpha
    blend = mask_crop[:, :, None] * ov_a * openness * ALPHA_IRIS

    result = ov_crop[:, :, :3].astype(np.float32) * blend + roi * (1.0 - blend)
    frame[fy1:fy2, fx1:fx2] = np.clip(result, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
#  Eyeshadow overlay
# ════════════════════════════════════════════════════════════
def draw_shadow(frame, lms, side, ov, ear_val):
    if ov is None:
        return
    openness = float(np.clip(
        (ear_val - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED), 0.0, 1.0))
    alpha = ALPHA_SHADOW * max(openness, 0.3)

    H, W   = frame.shape[:2]
    sh, sw = ov.shape[:2]

    if side == 'left':
        half = ov[:, :sw // 2]
        lx = int(lms[L_L].x * W);  rx = int(lms[L_R].x * W)
        ty = int(lms[L_BROW_OUT].y * H); by = int(lms[L_BOT].y * H)
    else:
        half = ov[:, sw // 2:]
        lx = int(lms[R_L].x * W);  rx = int(lms[R_R].x * W)
        ty = int(lms[R_BROW_OUT].y * H); by = int(lms[R_BOT].y * H)

    ew = abs(rx - lx); eh = max(abs(by - ty), 4)
    if ew < 4:
        return

    px = int(ew * 0.3); py = int(eh * 0.5)
    x1 = max(0, min(lx, rx) - px);  y1 = max(0, ty - py)
    x2 = min(W, max(lx, rx) + px);  y2 = min(H, by + int(eh * 0.2))
    rw, rh = x2 - x1, y2 - y1
    if rw < 2 or rh < 2:
        return

    resized = cv2.resize(half, (rw, rh), interpolation=cv2.INTER_AREA)
    roi = frame[y1:y2, x1:x2]

    ob = resized[:, :, :3].astype(np.float32)
    oa = (resized[:, :, 3:].astype(np.float32) / 255.0
          if resized.shape[2] == 4
          else np.ones((*resized.shape[:2], 1), np.float32)) * alpha
    frame[y1:y2, x1:x2] = np.clip(
        ob * oa + roi.astype(np.float32) * (1 - oa), 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
#  App
# ════════════════════════════════════════════════════════════
class EyeFilterApp:
    def __init__(self):
        print("\n👁️  EYE FILTER\n")
        mp_path = download_model("face_landmarker.task")
        self.det = _v.FaceLandmarker.create_from_options(
            _v.FaceLandmarkerOptions(
                base_options=_b.BaseOptions(model_asset_path=mp_path),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        print("✅ FaceLandmarker ready")

        iris_path  = next((f for f in ["eye_overlay.png","eye_overlay.jpg"] if os.path.exists(f)), "eye_overlay.png")
        shade_path = next((f for f in ["eyeshadow_only.png","eyeshadow_only.jpg","eyeshadow.png","eyeshadow.jpg"] if os.path.exists(f)), "eyeshadow_only.png")
        self.iris_ov  = load_overlay(iris_path,  "Iris sticker", bg="white")
        self.shade_ov = load_overlay(shade_path, "Eyeshadow",    bg="black")

        self.active             = False
        self.close_since        = None
        self.toggled_this_close = False

        print("\n📌 Close BOTH eyes for 3s → toggle filter ON/OFF")
        print("   S = reset sage mode   Q = quit\n")

    # ── blink-hold toggle ────────────────────────────────────
    def check_toggle(self, el, er, now):
        both = (el < EAR_CLOSED) and (er < EAR_CLOSED)
        if both:
            if self.close_since is None:
                self.close_since = now
            elif (now - self.close_since >= CLOSE_HOLD
                  and not self.toggled_this_close):
                self.active = not self.active
                self.toggled_this_close = True
                print(f"   Filter {'ON 👁️' if self.active else 'OFF'}")
        else:
            self.close_since        = None
            self.toggled_this_close = False

    # ── progress ring ────────────────────────────────────────
    def draw_ui(self, frame, now):
        if self.close_since is None or self.active:
            return
        elapsed = now - self.close_since
        pct = min(elapsed / CLOSE_HOLD, 1.0)
        rem = max(0.0, CLOSE_HOLD - elapsed)
        cx, cy, r = 55, 55, 40
        cv2.circle(frame, (cx, cy), r, (20, 20, 20), -1)
        if pct > 0:
            cv2.ellipse(frame, (cx, cy), (r-4, r-4),
                        -90, 0, int(pct*360), (0, 220, 200), 5, cv2.LINE_AA)
        lbl = f"{rem:.1f}s"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, lbl, (cx - tw//2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "CLOSE", (cx-18, cy+r+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,200), 1, cv2.LINE_AA)
        cv2.putText(frame, "EYES",  (cx-14, cy+r+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,200), 1, cv2.LINE_AA)

    # ── per-frame ────────────────────────────────────────────
    def process(self, frame):
        now = time.time()
        res = self.det.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if res.face_landmarks:
            lms = res.face_landmarks[0]
            el  = ear(lms, L_TOP, L_BOT, L_L, L_R)
            er  = ear(lms, R_TOP, R_BOT, R_L, R_R)

            # Debug EAR overlay so you can tune thresholds
            cv2.putText(frame,
                        f"EAR L:{el:.3f} R:{er:.3f}  closed<{EAR_CLOSED}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0, 200, 255), 1, cv2.LINE_AA)

            self.check_toggle(el, er, now)

            if self.active:
                # eyeshadow first (sits behind iris sticker)
                draw_shadow(frame, lms, 'left',  self.shade_ov, el)
                draw_shadow(frame, lms, 'right', self.shade_ov, er)

                # iris sticker with eyelid clip
                draw_iris_sticker(frame, lms,
                                  LEFT_IRIS,  L_TOP_ROW, L_BOT_ROW,
                                  self.iris_ov, el)
                draw_iris_sticker(frame, lms,
                                  RIGHT_IRIS, R_TOP_ROW, R_BOT_ROW,
                                  self.iris_ov, er)

        self.draw_ui(frame, now)

        col = (0, 220, 200) if self.active else (140, 140, 140)
        txt = "FILTER: ON  |  S = reset" if self.active else \
              "Close both eyes 3s to activate  |  S = reset"
        cv2.putText(frame, txt, (10, frame.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return frame

    # ── main loop ────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        for rw, rh in [(1280, 720), (854, 480), (640, 480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) >= rw - 10:
                break
        print(f"📷 Camera: {int(cap.get(3))}×{int(cap.get(4))}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.imshow("👁️  Eye Filter", self.process(frame))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                self.active             = False
                self.close_since        = None
                self.toggled_this_close = False
                print("🔄 Sage mode reset")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        EyeFilterApp().run()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        import traceback
        print(f"\n❌ {e}")
        traceback.print_exc()
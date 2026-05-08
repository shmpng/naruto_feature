"""
Eye Filter — Standalone
FILES NEEDED (same folder):
  eye_overlay.jpg      ← yellow minus circle  (white bg)
  eyeshadow_only.jpg   ← orange eyeshadow     (black bg)
  face_landmarker.task ← auto-downloaded on first run

INSTALL:  pip install mediapipe opencv-python numpy
KEYS:     Q = Quit   S = Reset filter   D = Debug overlay
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
LEFT_IRIS  = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

L_TOP, L_BOT = 159, 145
R_TOP, R_BOT = 386, 374
L_L,   L_R   = 33,  133
R_L,   R_R   = 362, 263

L_TOP_ROW = [33, 246, 161, 160, 159, 158, 157, 173, 133]
L_BOT_ROW = [33,   7, 163, 144, 145, 153, 154, 155, 133]
R_TOP_ROW = [362, 398, 384, 385, 386, 387, 388, 466, 263]
R_BOT_ROW = [362, 382, 381, 380, 374, 373, 390, 249, 263]

L_BROW_OUT = 46
R_BROW_OUT = 276

EAR_OPEN   = 0.22
EAR_CLOSED = 0.15
CLOSE_HOLD = 3.0
ALPHA_IRIS = 0.92


# ════════════════════════════════════════════════════════════
#  Image helpers  (defined BEFORE load_overlay)
# ════════════════════════════════════════════════════════════
def remove_white_bg(img):
    """Make white/near-white pixels transparent."""
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    mask = (b > 200) & (g > 200) & (r > 200)
    a = np.ones(img.shape[:2], np.uint8) * 255
    a[mask] = 0
    return cv2.merge([b, g, r, a])


def remove_black_bg(img):
    """Make black/near-black pixels transparent."""
    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
    mask = (b < 40) & (g < 40) & (r < 40)
    a = np.ones(img.shape[:2], np.uint8) * 255
    a[mask] = 0
    return cv2.merge([b, g, r, a])


def rotate_overlay(ov, angle_deg):
    """Rotate a BGRA image around its center, transparent fill."""
    h, w = ov.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle_deg, 1.0)
    return cv2.warpAffine(ov, M, (w, h),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0, 0))


def load_overlay(path, name, bg="white"):
    if not os.path.exists(path):
        print(f"⚠️  {name} not found at '{path}' — skipping")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  Cannot read {path}")
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = remove_white_bg(img) if bg == "white" else remove_black_bg(img)
    print(f"✅ Loaded {name}: {img.shape[1]}×{img.shape[0]}  "
          f"alpha non-zero: {np.count_nonzero(img[:,:,3])}")
    return img


def download_model(path):
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print(f"   Downloading {path} …")
        urllib.request.urlretrieve(url, path)
    return path


# ════════════════════════════════════════════════════════════
#  Face / iris helpers
# ════════════════════════════════════════════════════════════
def ear(lms, t, b, l, r):
    return abs(lms[t].y - lms[b].y) / (abs(lms[l].x - lms[r].x) + 1e-6)


def get_iris(lms, indices, w, h):
    """Returns (cx, cy, radius, angle_deg)."""
    cx = lms[indices[0]].x * w
    cy = lms[indices[0]].y * h
    radii = [np.hypot(lms[i].x*w - cx, lms[i].y*h - cy) for i in indices[1:]]
    radius = max(int(np.mean(radii)), 4)
    # angle from left-edge → right-edge of iris
    lx = lms[indices[1]].x * w;  ly = lms[indices[1]].y * h
    rx = lms[indices[3]].x * w;  ry = lms[indices[3]].y * h
    angle_deg = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    return int(cx), int(cy), radius, angle_deg


# ════════════════════════════════════════════════════════════
#  Iris sticker with eyelid clip + rotation
# ════════════════════════════════════════════════════════════
def draw_iris_sticker(frame, lms, iris_idx, top_row, bot_row, ov, ear_val):
    if ov is None:
        return

    openness = float(np.clip(
        (ear_val - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED), 0.0, 1.0))
    if openness < 0.02:
        return

    H, W = frame.shape[:2]
    cx, cy, radius, angle_deg = get_iris(lms, iris_idx, W, H)

    # Resize then rotate to match real eye axis
    scale   = 1.15
    diam    = max(int(radius * 2 * scale), 8)
    half    = diam // 2
    resized = cv2.resize(ov, (diam, diam), interpolation=cv2.INTER_AREA)
    resized = rotate_overlay(resized, angle_deg)

    # Build eyelid clipping polygon in sticker-local coords
    def lm_px(idx):
        return np.array([lms[idx].x * W, lms[idx].y * H])

    def to_local(px, py):
        return (int(round(px - (cx - half))),
                int(round(py - (cy - half))))

    upper = [to_local(*lm_px(i)) for i in top_row]
    lower = [to_local(*lm_px(i)) for i in reversed(bot_row)]
    poly  = np.array(upper + lower, dtype=np.int32)

    lid_mask = np.zeros((diam, diam), np.uint8)
    cv2.fillPoly(lid_mask, [poly], 255)

    # Eyelash inset — trims edge so sticker doesn't bleed past lash line
    inset  = max(1, int(radius * 0.08))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (inset*2+1, inset*2+1))
    lid_mask = cv2.erode(lid_mask, kernel, iterations=1)

    # Blit onto frame
    x1, y1 = cx - half, cy - half
    fx1 = max(0, x1);  fy1 = max(0, y1)
    fx2 = min(W, x1 + diam);  fy2 = min(H, y1 + diam)
    if fx2 <= fx1 or fy2 <= fy1:
        return

    ox1 = fx1 - x1;  oy1 = fy1 - y1
    ox2 = ox1 + (fx2 - fx1);  oy2 = oy1 + (fy2 - fy1)

    roi      = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    ov_crop  = resized[oy1:oy2, ox1:ox2]
    mask_crop= lid_mask[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0

    ov_a = (ov_crop[:,:,3:].astype(np.float32) / 255.0
            if ov_crop.shape[2] == 4
            else np.ones((*ov_crop.shape[:2], 1), np.float32))

    blend  = mask_crop[:,:,None] * ov_a * openness * ALPHA_IRIS
    result = ov_crop[:,:,:3].astype(np.float32) * blend + roi * (1.0 - blend)
    frame[fy1:fy2, fx1:fx2] = np.clip(result, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
#  App
# ════════════════════════════════════════════════════════════
class EyeFilterApp:
    def __init__(self):
        print("\n👁️  EYE FILTER\n")
        self.det = _v.FaceLandmarker.create_from_options(
            _v.FaceLandmarkerOptions(
                base_options=_b.BaseOptions(
                    model_asset_path=download_model("face_landmarker.task")),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        print("✅ FaceLandmarker ready")

        iris_path  = next((f for f in
                           ["eye_overlay.png", "eye_overlay.jpg"]
                           if os.path.exists(f)), "eye_overlay.png")
        self.iris_ov = load_overlay(iris_path, "Iris sticker", bg="white")

        self.active             = False
        self.close_since        = None
        self.toggled_this_close = False
        self.debug              = False

        print("\n📌 Close BOTH eyes for 3s → toggle filter ON/OFF")
        print("   S = reset   D = debug   Q = quit\n")

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
                        -90, 0, int(pct*360), (0,220,200), 5, cv2.LINE_AA)
        lbl = f"{rem:.1f}s"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, lbl, (cx-tw//2, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(frame, "CLOSE", (cx-18, cy+r+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,200), 1, cv2.LINE_AA)
        cv2.putText(frame, "EYES",  (cx-14, cy+r+30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,220,200), 1, cv2.LINE_AA)

    def process(self, frame):
        now = time.time()
        res = self.det.detect(mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

        if res.face_landmarks:
            lms = res.face_landmarks[0]
            el  = ear(lms, L_TOP, L_BOT, L_L, L_R)
            er  = ear(lms, R_TOP, R_BOT, R_L, R_R)

            cv2.putText(frame,
                        f"EAR L:{el:.3f} R:{er:.3f}  active:{self.active}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (0,200,255), 1, cv2.LINE_AA)

            self.check_toggle(el, er, now)

            if self.active or self.debug:
                draw_iris_sticker(frame, lms,
                                  LEFT_IRIS,  L_TOP_ROW, L_BOT_ROW,
                                  self.iris_ov, el)
                draw_iris_sticker(frame, lms,
                                  RIGHT_IRIS, R_TOP_ROW, R_BOT_ROW,
                                  self.iris_ov, er)

        self.draw_ui(frame, now)

        col = (0,220,200) if self.active else (140,140,140)
        txt = ("FILTER: ON  |  S=reset  D=debug"
               if self.active else
               "Close both eyes 3s to activate  |  S=reset  D=debug")
        cv2.putText(frame, txt, (10, frame.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return frame

    def run(self):
        cap = cv2.VideoCapture(0)
        for rw, rh in [(1280,720), (854,480), (640,480)]:
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
            if k in (ord('q'), ord('Q')):
                break
            if k in (ord('s'), ord('S')):
                self.active             = False
                self.close_since        = None
                self.toggled_this_close = False
                print("🔄 Sage mode reset")
            if k in (ord('d'), ord('D')):
                self.debug = not self.debug
                print(f"🔍 Debug: {'ON' if self.debug else 'OFF'}")

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
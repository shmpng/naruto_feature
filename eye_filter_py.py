"""
Eye Filter — Standalone
FILES NEEDED (same folder):
  eye_overlay.png      ← yellow minus circle
  eyeshadow.png        ← orange eyeshadow
  face_landmarker.task ← auto-downloaded on first run

INSTALL:  pip install mediapipe opencv-python numpy
KEYS:     Q = Quit   R = Reset filter
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

L_TOP, L_BOT, L_L, L_R = 159, 145,  33, 133
R_TOP, R_BOT, R_L, R_R = 386, 374, 362, 263

L_TOP_ROW = [157, 158, 159, 160, 161]
L_BOT_ROW = [144, 145, 153, 163, 173]
R_TOP_ROW = [384, 385, 386, 387, 388]
R_BOT_ROW = [380, 374, 373, 390, 249]

L_BROW_OUT = 46
R_BROW_OUT = 276

# ── Tuning constants ─────────────────────────────────────────
EAR_OPEN     = 0.22
EAR_CLOSED   = 0.13    # raised — most cameras need 0.10-0.15
CLOSE_HOLD   = 3.0
ALPHA_IRIS   = 0.90
ALPHA_SHADOW = 0.72


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════
def download_model(path):
    if not os.path.exists(path):
        url = ("https://storage.googleapis.com/mediapipe-models/"
               "face_landmarker/face_landmarker/float16/1/face_landmarker.task")
        print(f"   Downloading {path} …")
        urllib.request.urlretrieve(url, path)
    return path


def load_overlay(path, name):
    if not os.path.exists(path):
        print(f"⚠️  {name} not found at '{path}' — skipping"); return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  Cannot read {path}"); return None
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    print(f"✅ Loaded {name}: {img.shape[1]}×{img.shape[0]}")
    return img


def ear(lms, t, b, l, r):
    return abs(lms[t].y - lms[b].y) / (abs(lms[l].x - lms[r].x) + 1e-6)


def iris_info(lms, indices, w, h):
    cx = int(lms[indices[0]].x * w)
    cy = int(lms[indices[0]].y * h)
    radii = [np.hypot(lms[i].x*w - cx, lms[i].y*h - cy) for i in indices[1:]]
    return cx, cy, max(int(np.mean(radii)) if radii else 8, 4)


def alpha_blend(bg_roi, ov_rgba, mul=1.0):
    ob = ov_rgba[:,:,:3].astype(np.float32)
    oa = (ov_rgba[:,:,3:].astype(np.float32)/255.0 if ov_rgba.shape[2]==4
          else np.ones((*ov_rgba.shape[:2],1), np.float32)) * mul
    return np.clip(ob*oa + bg_roi.astype(np.float32)*(1-oa), 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
#  Iris sticker (eyelid-clipped)
# ════════════════════════════════════════════════════════════
def draw_iris(frame, lms, iris_idx, top_row, bot_row, t_idx, b_idx, ov, ear_val):
    if ov is None: return
    openness = np.clip((ear_val - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED), 0.0, 1.0)
    if openness < 0.05: return

    h, w = frame.shape[:2]
    cx, cy, radius = iris_info(lms, iris_idx, w, h)
    diam = radius * 2
    if diam < 6: return

    resized = cv2.resize(ov, (diam, diam), interpolation=cv2.INTER_AREA)

    # Build eyelid polygon in sticker-space
    def world(idxs):
        return sorted([(int(lms[i].x*w), int(lms[i].y*h)) for i in idxs], key=lambda p: p[0])

    def stk(px, py):
        return (np.clip(int(px-(cx-radius)), 0, diam-1),
                np.clip(int(py-(cy-radius)), 0, diam-1))

    poly = [stk(*p) for p in world(top_row)] + [stk(*p) for p in reversed(world(bot_row))]
    lid  = np.zeros((diam, diam), np.uint8)
    if len(poly) >= 3:
        cv2.fillPoly(lid, [np.array(poly, np.int32)], 255)
    else:
        uy = np.clip(int(lms[t_idx].y*h-(cy-radius)), 0, diam-1)
        ly = np.clip(int(lms[b_idx].y*h-(cy-radius)), 0, diam-1)
        if ly > uy: lid[uy:ly+1,:] = 255

    x1,y1 = cx-radius, cy-radius
    fx1,fy1 = max(0,x1), max(0,y1)
    fx2,fy2 = min(w,x1+diam), min(h,y1+diam)
    ox1,oy1 = fx1-x1, fy1-y1
    if fx2<=fx1 or fy2<=fy1: return

    roi  = frame[fy1:fy2, fx1:fx2]
    ovc  = resized[oy1:oy1+(fy2-fy1), ox1:ox1+(fx2-fx1)]
    lidc = lid[oy1:oy1+(fy2-fy1), ox1:ox1+(fx2-fx1)].astype(np.float32)[:,:,None]/255.0
    oa   = (ovc[:,:,3:].astype(np.float32)/255.0 if ovc.shape[2]==4
            else np.ones((*ovc.shape[:2],1), np.float32))

    blend  = lidc * oa * openness * ALPHA_IRIS
    result = ovc[:,:,:3].astype(np.float32)*blend + roi.astype(np.float32)*(1-blend)
    frame[fy1:fy2, fx1:fx2] = np.clip(result, 0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════
#  Eyeshadow overlay
# ════════════════════════════════════════════════════════════
def draw_shadow(frame, lms, side, ov, ear_val):
    if ov is None: return
    openness = np.clip((ear_val-EAR_CLOSED)/(EAR_OPEN-EAR_CLOSED), 0.0, 1.0)
    alpha    = ALPHA_SHADOW * max(openness, 0.3)

    h, w   = frame.shape[:2]
    sh, sw = ov.shape[:2]

    if side == 'left':
        half = ov[:, :sw//2]
        lx, rx = int(lms[L_L].x*w), int(lms[L_R].x*w)
        ty, by = int(lms[L_BROW_OUT].y*h), int(lms[L_BOT].y*h)
    else:
        half = ov[:, sw//2:]
        lx, rx = int(lms[R_L].x*w), int(lms[R_R].x*w)
        ty, by = int(lms[R_BROW_OUT].y*h), int(lms[R_BOT].y*h)

    ew = abs(rx-lx); eh = max(abs(by-ty), 4)
    if ew < 4: return

    px, py = int(ew*0.3), int(eh*0.5)
    x1 = max(0, min(lx,rx)-px);  y1 = max(0, ty-py)
    x2 = min(w, max(lx,rx)+px);  y2 = min(h, by+int(eh*0.2))
    rw, rh = x2-x1, y2-y1
    if rw<2 or rh<2: return

    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = alpha_blend(roi, cv2.resize(half,(rw,rh),interpolation=cv2.INTER_AREA), alpha)


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

        self.iris_ov  = load_overlay("eye_overlay.png", "Iris sticker")
        self.shade_ov = load_overlay("eyeshadow.png",   "Eyeshadow")

        self.active             = False
        self.close_since        = None
        self.toggled_this_close = False
        print("\n📌 Close BOTH eyes for 3s → toggle filter ON/OFF")
        print("   Q = Quit   R = Reset\n")

    # ── activation ──────────────────────────────────────────
    def check_toggle(self, el, er, now):
        both_closed = (el < EAR_CLOSED) and (er < EAR_CLOSED)
        if both_closed:
            if self.close_since is None:
                self.close_since = now
            elif now - self.close_since >= CLOSE_HOLD and not self.toggled_this_close:
                self.active = not self.active
                self.toggled_this_close = True
                print(f"   Filter {'ON 👁️' if self.active else 'OFF'}")
        else:
            self.close_since = None
            self.toggled_this_close = False

    # ── progress ring ────────────────────────────────────────
    def draw_ui(self, frame, now):
        if self.close_since is None or self.active: return
        pct = min((now-self.close_since)/CLOSE_HOLD, 1.0)
        rem = max(0, CLOSE_HOLD-(now-self.close_since))
        cx, cy, r = 55, 55, 40
        cv2.circle(frame, (cx,cy), r, (20,20,20), -1)
        if pct > 0:
            cv2.ellipse(frame,(cx,cy),(r-4,r-4),-90,0,int(pct*360),(0,220,200),5,cv2.LINE_AA)
        lbl = f"{rem:.1f}s"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, lbl, (cx-tw//2, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.44,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,"CLOSE",(cx-18,cy+r+16),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,200),1,cv2.LINE_AA)
        cv2.putText(frame,"EYES", (cx-14,cy+r+30),cv2.FONT_HERSHEY_SIMPLEX,0.38,(0,220,200),1,cv2.LINE_AA)

    # ── per-frame ────────────────────────────────────────────
    def process(self, frame):
        now  = time.time()
        h, w = frame.shape[:2]
        res  = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        if res.face_landmarks:
            lms  = res.face_landmarks[0]
            el   = ear(lms, L_TOP, L_BOT, L_L, L_R)
            er   = ear(lms, R_TOP, R_BOT, R_L, R_R)
            # Debug: print EAR values so you can tune thresholds
            cv2.putText(frame, f"EAR L:{el:.3f} R:{er:.3f} (closed<{EAR_CLOSED})",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0,200,255), 1, cv2.LINE_AA)
            self.check_toggle(el, er, now)
            if self.active:
                draw_shadow(frame, lms, 'left',  self.shade_ov, el)
                draw_shadow(frame, lms, 'right', self.shade_ov, er)
                draw_iris(frame, lms, LEFT_IRIS,  L_TOP_ROW, L_BOT_ROW, L_TOP, L_BOT, self.iris_ov, el)
                draw_iris(frame, lms, RIGHT_IRIS, R_TOP_ROW, R_BOT_ROW, R_TOP, R_BOT, self.iris_ov, er)

        self.draw_ui(frame, now)
        col = (0,220,200) if self.active else (140,140,140)
        txt = "FILTER: ON  |  S = reset" if self.active else "Close both eyes 3s to activate  |  S = reset"
        cv2.putText(frame, txt, (10, frame.shape[0]-14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return frame

    # ── main loop ────────────────────────────────────────────
    def run(self):
        cap = cv2.VideoCapture(0)
        for rw, rh in [(1280,720),(854,480),(640,480)]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, rw)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rh)
            if int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) >= rw-10: break
        print(f"📷 Camera: {int(cap.get(3))}×{int(cap.get(4))}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            cv2.imshow("👁️  Eye Filter", self.process(frame))
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'): break
            if k in (ord('s'),):
                self.active = False; self.close_since = None
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
        import traceback; print(f"\n❌ {e}"); traceback.print_exc()
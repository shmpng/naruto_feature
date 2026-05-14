"""
app.py
SageEyeFilterApp — orchestrates face detection, the close-both-eyes
toggle, and rendering of both filters every frame.
"""

import cv2
import numpy as np
import os
import time

import mediapipe as mp
from mediapipe.tasks.python import vision as _v
from mediapipe.tasks.python.core import base_options as _b

from landmarks import (
    LEFT_IRIS, RIGHT_IRIS,
    L_TOP, L_BOT, R_TOP, R_BOT,
    L_L, L_R, R_L, R_R,
    L_TOP_ROW, L_BOT_ROW, R_TOP_ROW, R_BOT_ROW,
    R_INNER, R_OUTER, L_INNER, L_OUTER,
    R_UPPER_LID, R_LOWER_LID, L_UPPER_LID, L_LOWER_LID,
    NORM_SOCKET_RIGHT, NORM_SOCKET_LEFT,
)
from utils import (
    EAR_CLOSED, CLOSE_HOLD,
    download_model, load_iris_overlay, ear,
)
from iris_filter import draw_iris_sticker
from eyeshadow_filter import apply_sage_eye


class SageEyeFilterApp:
    def __init__(self):
        print("\n👁️  SAGE MODE + IRIS FILTER\n")

        # ── MediaPipe face landmarker ─────────────────────────────
        self.det = _v.FaceLandmarker.create_from_options(
            _v.FaceLandmarkerOptions(
                base_options=_b.BaseOptions(
                    model_asset_path=download_model("face_landmarker.task")
                ),
                output_face_blendshapes=True,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        )
        print("✅ FaceLandmarker ready")

        # ── Iris sticker ──────────────────────────────────────────
        iris_path = next(
            (f for f in ["iris.png", "iris.jpg"] if os.path.exists(f)),
            "iris.jpg",
        )
        self.iris_ov = load_iris_overlay(iris_path)

        # ── Orange eyeshadow ──────────────────────────────────────
        orange_path = "eyeshadow.jpg"
        if os.path.exists(orange_path):
            orange = cv2.imread(orange_path)
            self.orange      = orange
            self.orange_flip = cv2.flip(orange, 1)   # mirrored for left eye
            print(f"✅ Orange eyeshadow: {orange.shape[1]}×{orange.shape[0]}")
        else:
            print(f"⚠️  '{orange_path}' not found — eyeshadow disabled")
            self.orange = self.orange_flip = None

        # ── Toggle state ──────────────────────────────────────────
        self.active             = False
        self.close_since        = None
        self.toggled_this_close = False
        self.debug              = False

        print("\n📌 Close BOTH eyes for 3 s  →  toggle filters ON / OFF")
        print("   S = reset   D = debug iris   Q = quit\n")

    # ─────────────────────────────────────────────────────────────
    def _check_toggle(self, el: float, er: float, now: float) -> None:
        """Toggle active state when both eyes have been closed ≥ CLOSE_HOLD s."""
        both_closed = (el < EAR_CLOSED) and (er < EAR_CLOSED)
        if both_closed:
            if self.close_since is None:
                self.close_since = now
            elif now - self.close_since >= CLOSE_HOLD and not self.toggled_this_close:
                self.active = not self.active
                self.toggled_this_close = True
                print(f"   Filters {'ON 👁️' if self.active else 'OFF'}")
        else:
            self.close_since        = None
            self.toggled_this_close = False

    # ─────────────────────────────────────────────────────────────
    def _draw_countdown_ui(self, frame: np.ndarray, now: float) -> None:
        """Teal progress ring in the top-left while the user is blinking."""
        if self.close_since is None or self.active:
            return
        elapsed = now - self.close_since
        pct = min(elapsed / CLOSE_HOLD, 1.0)
        rem = max(0.0, CLOSE_HOLD - elapsed)
        cx, cy, r = 55, 55, 40
        cv2.circle(frame, (cx, cy), r, (20, 20, 20), -1)
        if pct > 0:
            cv2.ellipse(frame, (cx, cy), (r - 4, r - 4),
                        -90, 0, int(pct * 360), (0, 220, 200), 5, cv2.LINE_AA)
        lbl = f"{rem:.1f}s"
        (tw, _), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.putText(frame, lbl,    (cx - tw // 2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "CLOSE", (cx - 18, cy + r + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, "EYES",  (cx - 14, cy + r + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 200), 1, cv2.LINE_AA)

    # ─────────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run detection + filters on a single BGR frame. Returns the frame."""
        now  = time.time()
        H, W = frame.shape[:2]

        res = self.det.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            )
        )

        if res.face_landmarks:
            lms = res.face_landmarks[0]
            el  = ear(lms, L_TOP, L_BOT, L_L, L_R)
            er  = ear(lms, R_TOP, R_BOT, R_L, R_R)

            cv2.putText(
                frame,
                f"EAR L:{el:.3f} R:{er:.3f}  active:{self.active}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 200, 255), 1, cv2.LINE_AA,
            )

            self._check_toggle(el, er, now)

            if self.active or self.debug:
                # 1. Orange eyeshadow — drawn first (sits under the iris sticker)
                if self.orange is not None:
                    apply_sage_eye(
                        frame, lms, W, H, self.orange,
                        inner_idx=R_INNER, outer_idx=R_OUTER,
                        upper_lid_idx=R_UPPER_LID, lower_lid_idx=R_LOWER_LID,
                        norm_socket=NORM_SOCKET_RIGHT,
                    )
                    apply_sage_eye(
                        frame, lms, W, H, self.orange_flip,
                        inner_idx=L_INNER, outer_idx=L_OUTER,
                        upper_lid_idx=L_UPPER_LID, lower_lid_idx=L_LOWER_LID,
                        norm_socket=NORM_SOCKET_LEFT,
                    )

                # 2. Iris sticker — drawn on top
                draw_iris_sticker(frame, lms, LEFT_IRIS,  L_TOP_ROW, L_BOT_ROW,
                                  self.iris_ov, el)
                draw_iris_sticker(frame, lms, RIGHT_IRIS, R_TOP_ROW, R_BOT_ROW,
                                  self.iris_ov, er)

        self._draw_countdown_ui(frame, now)

        col = (0, 220, 200) if self.active else (140, 140, 140)
        txt = (
            "SAGE MODE: ON  |  S=reset  D=debug  Q=quit"
            if self.active else
            "Close both eyes 3 s to activate  |  S=reset  D=debug  Q=quit"
        )
        cv2.putText(frame, txt, (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
        return frame

    # ─────────────────────────────────────────────────────────────
    def run(self) -> None:
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
            cv2.imshow("👁️  Sage Mode + Iris Filter", self.process(frame))

            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), ord("Q")):
                break
            if k in (ord("s"), ord("S")):
                self.active = self.toggled_this_close = False
                self.close_since = None
                print("🔄 Filters reset")
            if k in (ord("d"), ord("D")):
                self.debug = not self.debug
                print(f"🔍 Debug: {'ON' if self.debug else 'OFF'}")

        cap.release()
        cv2.destroyAllWindows()

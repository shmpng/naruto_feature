"""
utils.py
Shared helper functions used across the Sage Mode + Iris filters.
"""

import cv2
import numpy as np
import os
import urllib.request


# ── Toggle / blink constants ──────────────────────────────────────────────────
EAR_OPEN   = 0.22   # eye fully open threshold
EAR_CLOSED = 0.15   # eye fully closed threshold
CLOSE_HOLD = 3.0    # seconds both eyes must stay closed to toggle
ALPHA_IRIS = 0.92   # iris sticker max opacity


# ── MediaPipe model ───────────────────────────────────────────────────────────
def download_model(path: str) -> str:
    """Download face_landmarker.task if not already present."""
    if not os.path.exists(path):
        url = (
            "https://storage.googleapis.com/mediapipe-models/"
            "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        )
        print(f"   Downloading {path} …")
        urllib.request.urlretrieve(url, path)
    return path


# ── Image helpers ─────────────────────────────────────────────────────────────
def remove_white_bg(img: np.ndarray) -> np.ndarray:
    """Make white / near-white pixels transparent (returns BGRA)."""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    mask = (b > 200) & (g > 200) & (r > 200)
    a = np.ones(img.shape[:2], np.uint8) * 255
    a[mask] = 0
    return cv2.merge([b, g, r, a])


def rotate_overlay(ov: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a BGRA image around its center with transparent fill."""
    h, w = ov.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), -angle_deg, 1.0)
    return cv2.warpAffine(
        ov, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def load_iris_overlay(path: str) -> np.ndarray | None:
    """Load iris sticker, strip white background → BGRA. Returns None on failure."""
    if not os.path.exists(path):
        print(f"⚠️  Iris overlay not found at '{path}' — skipping")
        return None
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  Cannot read '{path}'")
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    img = remove_white_bg(img)
    print(
        f"✅ Iris overlay: {img.shape[1]}×{img.shape[0]}  "
        f"alpha non-zero: {np.count_nonzero(img[:, :, 3])}"
    )
    return img


# ── Landmark helpers ──────────────────────────────────────────────────────────
def ear(lms, t: int, b: int, l: int, r: int) -> float:
    """Eye Aspect Ratio — proxy for how open the eye is."""
    return abs(lms[t].y - lms[b].y) / (abs(lms[l].x - lms[r].x) + 1e-6)


def get_iris(lms, indices: list[int], w: int, h: int) -> tuple[int, int, int, float]:
    """
    Returns (cx, cy, radius, angle_deg) for an iris cluster.
    indices[0] = centre, indices[1..4] = rim points.
    """
    cx = lms[indices[0]].x * w
    cy = lms[indices[0]].y * h
    radii = [np.hypot(lms[i].x * w - cx, lms[i].y * h - cy) for i in indices[1:]]
    radius = max(int(np.mean(radii)), 4)
    lx = lms[indices[1]].x * w;  ly = lms[indices[1]].y * h
    rx = lms[indices[3]].x * w;  ry = lms[indices[3]].y * h
    angle_deg = float(np.degrees(np.arctan2(ry - ly, rx - lx)))
    return int(cx), int(cy), radius, angle_deg


def lm_px(lms, idx: int, W: int, H: int) -> np.ndarray:
    """Convert a single normalized landmark to pixel coordinates."""
    return np.array([lms[idx].x * W, lms[idx].y * H], dtype=np.float32)

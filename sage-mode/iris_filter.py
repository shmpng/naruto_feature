"""
iris_filter.py
Draws the yellow iris sticker over each eye, clipped to the real eyelid shape
and faded proportionally to how open the eye is.
"""

import cv2
import numpy as np

from utils import (
    EAR_OPEN, EAR_CLOSED, ALPHA_IRIS,
    rotate_overlay, get_iris,
)


def draw_iris_sticker(
    frame: np.ndarray,
    lms,
    iris_idx: list[int],
    top_row: list[int],
    bot_row: list[int],
    ov: np.ndarray | None,
    ear_val: float,
) -> None:
    """
    Blend the iris overlay onto `frame` for one eye.

    Parameters
    ----------
    frame    : BGR frame (modified in-place)
    lms      : MediaPipe face landmark list
    iris_idx : 5 iris landmark indices [centre, rim×4]
    top_row  : upper eyelid landmark indices (left → right)
    bot_row  : lower eyelid landmark indices (left → right)
    ov       : BGRA iris overlay image, or None to skip
    ear_val  : pre-computed Eye Aspect Ratio for this eye
    """
    if ov is None:
        return

    openness = float(
        np.clip((ear_val - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED), 0.0, 1.0)
    )
    if openness < 0.02:
        return

    H, W = frame.shape[:2]
    cx, cy, radius, angle_deg = get_iris(lms, iris_idx, W, H)

    # Resize sticker and rotate to match eye tilt
    scale   = 1.15
    diam    = max(int(radius * 2 * scale), 8)
    half    = diam // 2
    resized = cv2.resize(ov, (diam, diam), interpolation=cv2.INTER_AREA)
    resized = rotate_overlay(resized, angle_deg)

    # Build eyelid clipping polygon in sticker-local coordinates
    def lm_local(idx):
        return np.array([lms[idx].x * W, lms[idx].y * H])

    def to_local(px, py):
        return (int(round(px - (cx - half))), int(round(py - (cy - half))))

    upper = [to_local(*lm_local(i)) for i in top_row]
    lower = [to_local(*lm_local(i)) for i in reversed(bot_row)]
    poly  = np.array(upper + lower, dtype=np.int32)

    lid_mask = np.zeros((diam, diam), np.uint8)
    cv2.fillPoly(lid_mask, [poly], 255)

    # Erode slightly so sticker doesn't bleed past the lash line
    inset  = max(1, int(radius * 0.08))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)
    )
    lid_mask = cv2.erode(lid_mask, kernel, iterations=1)

    # Clip blit region to frame bounds
    x1, y1 = cx - half, cy - half
    fx1 = max(0, x1);  fy1 = max(0, y1)
    fx2 = min(W, x1 + diam);  fy2 = min(H, y1 + diam)
    if fx2 <= fx1 or fy2 <= fy1:
        return

    ox1 = fx1 - x1;  oy1 = fy1 - y1
    ox2 = ox1 + (fx2 - fx1);  oy2 = oy1 + (fy2 - fy1)

    roi       = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    ov_crop   = resized[oy1:oy2, ox1:ox2]
    mask_crop = lid_mask[oy1:oy2, ox1:ox2].astype(np.float32) / 255.0

    ov_a = (
        ov_crop[:, :, 3:].astype(np.float32) / 255.0
        if ov_crop.shape[2] == 4
        else np.ones((*ov_crop.shape[:2], 1), np.float32)
    )

    blend  = mask_crop[:, :, None] * ov_a * openness * ALPHA_IRIS
    result = ov_crop[:, :, :3].astype(np.float32) * blend + roi * (1.0 - blend)
    frame[fy1:fy2, fx1:fx2] = np.clip(result, 0, 255).astype(np.uint8)

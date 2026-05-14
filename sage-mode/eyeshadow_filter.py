"""
eyeshadow_filter.py
Applies the orange Sage Mode eyeshadow to one eye at a time.
The colour fills the socket area above the open-eyelid line and is
clipped so it never bleeds below the lower lash arc.
Black lash lines are drawn on top.
"""

import cv2
import numpy as np

from utils import lm_px


def apply_sage_eye(
    frame: np.ndarray,
    lms,
    W: int,
    H: int,
    orange_img: np.ndarray,
    inner_idx: int,
    outer_idx: int,
    upper_lid_idx: list[int],
    lower_lid_idx: list[int],
    norm_socket: list[tuple[float, float]],
) -> None:
    """
    Paint orange eyeshadow onto `frame` for one eye (modified in-place).

    Parameters
    ----------
    frame         : BGR frame
    lms           : MediaPipe face landmark list
    W, H          : frame dimensions
    orange_img    : BGR source texture (BGR, no alpha needed)
    inner_idx     : landmark index for the eye's inner corner
    outer_idx     : landmark index for the eye's outer corner
    upper_lid_idx : upper eyelid landmark indices
    lower_lid_idx : lower eyelid landmark indices
    norm_socket   : pre-normalised eye-socket polygon (units of eye_width)
    """
    inner_pt  = lm_px(lms, inner_idx, W, H)
    outer_pt  = lm_px(lms, outer_idx, W, H)
    eye_width = np.linalg.norm(outer_pt - inner_pt)
    if eye_width < 4:
        return

    eye_cx = int((inner_pt[0] + outer_pt[0]) / 2)
    eye_cy = int((inner_pt[1] + outer_pt[1]) / 2)

    # Scale the socket polygon to the current eye width
    socket_poly = np.array(
        [
            [eye_cx + int(dx * eye_width), eye_cy + int(dy * eye_width)]
            for dx, dy in norm_socket
        ],
        dtype=np.int32,
    )

    upper_lid_pts = np.array(
        [[int(lms[i].x * W), int(lms[i].y * H)] for i in upper_lid_idx],
        dtype=np.int32,
    )
    lower_lid_pts = np.array(
        [[int(lms[i].x * W), int(lms[i].y * H)] for i in lower_lid_idx],
        dtype=np.int32,
    )

    # Region where the eye is open — exclude this from the paint area
    eye_opening_poly = np.vstack([upper_lid_pts, lower_lid_pts[::-1]])

    bx1 = max(0, int(socket_poly[:, 0].min()) - 4)
    by1 = max(0, int(socket_poly[:, 1].min()) - 4)
    bx2 = min(W, int(socket_poly[:, 0].max()) + 4)
    by2 = min(H, int(socket_poly[:, 1].max()) + 4)
    box_w = bx2 - bx1
    box_h = by2 - by1
    if box_w < 4 or box_h < 4:
        return

    # Scale orange texture to fit the bounding box width
    oh, ow = orange_img.shape[:2]
    scale    = box_w / ow
    scaled_w = box_w
    scaled_h = max(int(oh * scale), box_h)
    orange_s = cv2.resize(orange_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

    # Anchor the texture bottom to the socket bottom
    eye_bottom = int(socket_poly[:, 1].max())
    oy_start   = eye_bottom - scaled_h
    fy1 = max(0, oy_start);  fy2 = min(H, eye_bottom)
    fx1 = bx1;               fx2 = min(W, bx1 + scaled_w)

    sy1 = max(0, fy1 - oy_start)
    sy2 = min(scaled_h, sy1 + (fy2 - fy1))
    sx1 = 0
    sx2 = min(scaled_w, fx2 - fx1)
    fy2 = fy1 + (sy2 - sy1)
    fx2 = fx1 + (sx2 - sx1)
    if fy2 <= fy1 or fx2 <= fx1:
        return

    orange_patch = orange_s[sy1:sy2, sx1:sx2, :3]

    # Build composite mask: socket area MINUS the open-eye polygon,
    # further clipped to above the lower lash arc
    socket_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(socket_mask, [socket_poly], 255)

    eye_open_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(eye_open_mask, [eye_opening_poly], 255)

    lower_sorted = lower_lid_pts[np.argsort(lower_lid_pts[:, 0])]
    above_lower_poly = np.array(
        [[0, 0], [W, 0]] + lower_sorted[::-1].tolist(), dtype=np.int32
    )
    above_lower_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(above_lower_mask, [above_lower_poly], 255)

    paint_mask = cv2.bitwise_and(socket_mask, cv2.bitwise_not(eye_open_mask))
    paint_mask = cv2.bitwise_and(paint_mask, above_lower_mask)

    ry1 = max(0, min(by1, fy1));  ry2 = min(H, max(by2, fy2))
    rx1 = max(0, bx1);            rx2 = min(W, bx2)
    if ry2 <= ry1 or rx2 <= rx1:
        return

    roi      = frame[ry1:ry2, rx1:rx2].astype(np.float32)
    mask_roi = paint_mask[ry1:ry2, rx1:rx2].astype(np.float32) / 255.0

    canvas = np.zeros_like(roi)
    py1 = max(0, fy1 - ry1);  py2 = min(ry2 - ry1, py1 + (fy2 - fy1))
    px1 = max(0, fx1 - rx1);  px2 = min(rx2 - rx1, px1 + (fx2 - fx1))
    ph, pw = py2 - py1, px2 - px1
    if ph > 0 and pw > 0:
        canvas[py1:py2, px1:px2] = orange_patch[:ph, :pw]

    alpha   = mask_roi[:, :, None] * 0.93
    blended = canvas * alpha + roi * (1.0 - alpha)
    frame[ry1:ry2, rx1:rx2] = np.clip(blended, 0, 255).astype(np.uint8)

    # Draw solid black lash lines over the eyeshadow
    lash_t = max(1, int(eye_width * 0.035))
    cv2.polylines(frame, [upper_lid_pts], isClosed=False,
                  color=(0, 0, 0), thickness=lash_t, lineType=cv2.LINE_AA)
    cv2.polylines(frame, [lower_lid_pts], isClosed=False,
                  color=(0, 0, 0), thickness=lash_t, lineType=cv2.LINE_AA)

"""ROI helpers for sign-language inference.

The helpers try MediaPipe first when available, then fall back to a simple
skin/contour based hand detector, and finally to a center crop.
"""

from __future__ import annotations

import cv2
import numpy as np


def clamp_bbox(x, y, w, h, frame_shape):
    height, width = frame_shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def center_crop_bbox(frame_bgr, crop_ratio=0.75):
    height, width = frame_bgr.shape[:2]
    crop_w = max(1, int(width * crop_ratio))
    crop_h = max(1, int(height * crop_ratio))
    x = max(0, (width - crop_w) // 2)
    y = max(0, (height - crop_h) // 2)
    return clamp_bbox(x, y, crop_w, crop_h, frame_bgr.shape)


def expand_bbox(x, y, w, h, frame_shape, pad_ratio=0.25):
    pad_x = int(w * pad_ratio)
    pad_y = int(h * pad_ratio)
    return clamp_bbox(x - pad_x, y - pad_y, w + 2 * pad_x, h + 2 * pad_y, frame_shape)


def detect_skin_bbox(frame_bgr, min_area_ratio=0.01):
    """Detect a likely hand region using a simple skin-color contour mask."""
    height, width = frame_bgr.shape[:2]
    frame_area = height * width

    blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)

    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best)
    if area < frame_area * min_area_ratio:
        return None

    x, y, w, h = cv2.boundingRect(best)
    return expand_bbox(x, y, w, h, frame_bgr.shape, pad_ratio=0.30)


def detect_hand_roi(frame_bgr, mp_hands=None, use_mediapipe=True, crop_ratio=0.75):
    """Return (bbox, source) for the best available hand ROI."""
    if use_mediapipe and mp_hands is not None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(frame_rgb)
        if results and results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            height, width = frame_bgr.shape[:2]
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * width)
            x_max = int(max(x_coords) * width)
            y_min = int(min(y_coords) * height)
            y_max = int(max(y_coords) * height)
            x, y, w, h = expand_bbox(x_min, y_min, x_max - x_min, y_max - y_min, frame_bgr.shape, pad_ratio=0.20)
            return (x, y, w, h), "mediapipe"

    skin_bbox = detect_skin_bbox(frame_bgr)
    if skin_bbox is not None:
        return skin_bbox, "skin"

    return center_crop_bbox(frame_bgr, crop_ratio=crop_ratio), "center"

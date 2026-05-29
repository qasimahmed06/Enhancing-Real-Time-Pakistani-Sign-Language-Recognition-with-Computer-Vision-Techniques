"""Shared frame preprocessing utilities."""

import cv2
import numpy as np


def apply_canny_edge_enhancement(image, low_threshold=100, high_threshold=200, blend_ratio=0.2):
    """Blend a Canny edge map into an RGB image without changing its shape."""
    if image is None:
        return image

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    enhanced = cv2.addWeighted(image, 1.0 - blend_ratio, edges_rgb, blend_ratio, 0)
    return enhanced

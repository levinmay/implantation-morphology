"""Largest-component cleanup with bounded hole filling."""

import cv2
import numpy as np

HOLE_PARAMETERS = {"area_fraction": 0.001, "minimum_pixels": 16, "maximum_pixels": 2048}


def largest_component(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or 0 in mask.shape:
        raise ValueError("Expected a nonzero-size two-dimensional mask")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == label

def fill_tiny_holes(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask.astype(bool)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, 8)
    result = mask.astype(bool).copy()
    scaled_area = int(round(int(result.sum()) * HOLE_PARAMETERS["area_fraction"]))
    limit = max(HOLE_PARAMETERS["minimum_pixels"],
                min(HOLE_PARAMETERS["maximum_pixels"], scaled_area))
    height, width = result.shape
    for label in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[label])
        if x > 0 and y > 0 and x + w < width and y + h < height and area <= limit:
            result[labels == label] = True
    return result

def topology(mask: np.ndarray) -> np.ndarray:
    return fill_tiny_holes(largest_component(mask))

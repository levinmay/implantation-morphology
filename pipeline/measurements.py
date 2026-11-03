"""Area and external-contour circularity without changing staging features."""

import math
import cv2
import numpy as np


def mask_measurements(cleaned_mask: np.ndarray) -> dict:
    value = np.asarray(cleaned_mask, dtype=bool).astype(np.uint8)
    if value.ndim != 2:
        raise ValueError("Expected a two-dimensional binary mask")
    area = int(value.sum())
    contours, _ = cv2.findContours(value, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(cv2.arcLength(max(contours, key=cv2.contourArea), True)) if contours else 0.0
    circularity = min(1.0, 4 * math.pi * area / max(perimeter ** 2, 1e-8)) if area >= 3 else np.nan
    return {"cleaned_perimeter_px": perimeter, "cleaned_circularity": circularity}

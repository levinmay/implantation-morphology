"""The nine appearance features used in the paper.

``texture_glcm_energy`` is angular second moment (sum of squared
co-occurrence probabilities). ``texture_lbp_uniform_fraction`` uses the ten
codes defined below, not all conventional uniform-LBP patterns.
"""

import cv2
import numpy as np

MAX_TEXTURE_SIDE = 384

def normalized_image(image: np.ndarray) -> np.ndarray:
    result = np.empty_like(image, dtype=np.float32)
    for channel in range(3):
        plane = image[..., channel].astype(np.float32)
        low, high = np.quantile(plane, (0.01, 0.99))
        if high <= low:
            result[..., channel] = 0.0
        else:
            result[..., channel] = np.clip((plane - low) / (high - low), 0.0, 1.0)
    return result

def glcm_features(gray: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    quantized = np.minimum((gray * 16).astype(np.int16), 15)
    matrix = np.zeros((16, 16), dtype=np.float64)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        y0a, y1a = max(0, -dy), gray.shape[0] - max(0, dy)
        x0a, x1a = max(0, -dx), gray.shape[1] - max(0, dx)
        y0b, y1b = max(0, dy), gray.shape[0] - max(0, -dy)
        x0b, x1b = max(0, dx), gray.shape[1] - max(0, -dx)
        valid = mask[y0a:y1a, x0a:x1a] & mask[y0b:y1b, x0b:x1b]
        first = quantized[y0a:y1a, x0a:x1a][valid]
        second = quantized[y0b:y1b, x0b:x1b][valid]
        np.add.at(matrix, (first, second), 1)
        np.add.at(matrix, (second, first), 1)
    if matrix.sum() == 0:
        return {f"texture_glcm_{name}": float("nan") for name in ("homogeneity", "energy", "entropy")}
    probability = matrix / matrix.sum()
    i, j = np.indices(matrix.shape)
    positive = probability[probability > 0]
    return {
        "texture_glcm_homogeneity": float((probability / (1.0 + (i - j) ** 2)).sum()),
        "texture_glcm_energy": float((probability ** 2).sum()),
        "texture_glcm_entropy": float(-(positive * np.log2(positive)).sum()),
    }


def texture_features(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    height, width = mask.shape
    scale = min(1.0, MAX_TEXTURE_SIDE / max(height, width))
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    normalized = normalized_image(image)
    gray = cv2.cvtColor(normalized, cv2.COLOR_RGB2GRAY)
    pixels = int(mask.sum())
    features: dict[str, float] = {}
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    features["texture_gradient_std"] = float(np.std(gradient[mask])) if pixels else np.nan

    code = np.zeros_like(gray, dtype=np.uint8)
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2], gray[:-2, 1:-1], gray[:-2, 2:], gray[1:-1, 2:],
        gray[2:, 2:], gray[2:, 1:-1], gray[2:, :-2], gray[1:-1, :-2],
    )
    for bit, neighbor in enumerate(neighbors):
        code[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << bit)
    lbp = code[mask]
    if lbp.size:
        features["texture_lbp_mean"] = float(lbp.mean() / 255.0)
        features["texture_lbp_uniform_fraction"] = float(np.mean(np.isin(lbp, (0, 255, 1, 2, 4, 8, 16, 32, 64, 128))))
    else:
        features.update({"texture_lbp_mean": np.nan, "texture_lbp_uniform_fraction": np.nan})
    features.update(glcm_features(gray, mask))

    kernel = np.ones((7, 7), np.uint8)
    inside_band = mask & ~(cv2.erode(mask.astype(np.uint8), kernel) > 0)
    outside_band = (cv2.dilate(mask.astype(np.uint8), kernel) > 0) & ~mask
    for name, plane in (("gray", gray), ("gradient", gradient)):
        inside = plane[inside_band]
        outside = plane[outside_band]
        features[f"texture_boundary_{name}_contrast"] = (
            float(inside.mean() - outside.mean()) if inside.size and outside.size else np.nan
        )
        if name == "gray":
            features["texture_boundary_gray_absolute_contrast"] = (
                float(abs(inside.mean() - outside.mean())) if inside.size and outside.size else np.nan
            )
    return features

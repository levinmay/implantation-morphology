"""The 49 morphology and 42 appearance candidates used for feature selection."""
from __future__ import annotations
import math
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.stats import kurtosis, skew
from skimage.morphology import skeletonize

MAX_TEXTURE_SIDE = 384

def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def aligned_bitmap(mask: np.ndarray, side: int = 16) -> np.ndarray:
    points = np.argwhere(mask)
    if len(points) < 3:
        return np.zeros((side, side), dtype=np.float32)
    xy = points[:, [1, 0]].astype(float)
    center = xy.mean(axis=0)
    covariance = np.cov(xy, rowvar=False, bias=True)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    ratio = math.sqrt(max(values.max(), 1e-9) / max(values.min(), 1e-9))
    angle = math.degrees(math.atan2(major[1], major[0])) if ratio >= 1.10 else 0.0
    matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    rotated = cv2.warpAffine(
        mask.astype(np.uint8), matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    y, x = np.where(rotated)
    if len(y) == 0:
        return np.zeros((side, side), dtype=np.float32)
    crop = rotated[y.min():y.max() + 1, x.min():x.max() + 1]
    available = side - 2
    scale = min(available / crop.shape[0], available / crop.shape[1])
    target = (
        max(1, int(round(crop.shape[1] * scale))),
        max(1, int(round(crop.shape[0] * scale))),
    )
    resized = cv2.resize(crop.astype(np.uint8), target, interpolation=cv2.INTER_AREA).astype(np.float32)
    canvas = np.zeros((side, side), dtype=np.float32)
    y0 = (side - resized.shape[0]) // 2
    x0 = (side - resized.shape[1]) // 2
    canvas[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
    return canvas


def ellipse_iou(mask: np.ndarray, contour: np.ndarray) -> float:
    if len(contour) < 5:
        return 0.0
    ellipse = cv2.fitEllipse(contour)
    candidate = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(candidate, ellipse, 1, -1)
    union = np.logical_or(mask, candidate).sum()
    return float(np.logical_and(mask, candidate).sum() / max(union, 1))


def radial_features(mask: np.ndarray) -> dict[str, float]:
    points = np.argwhere(mask)
    xy = points[:, [1, 0]].astype(float)
    center = xy.mean(axis=0)
    contour_list, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contour_list, key=cv2.contourArea).reshape(-1, 2).astype(float)
    delta = contour - center
    radii = np.linalg.norm(delta, axis=1)
    equivalent = max(math.sqrt(float(mask.sum()) / math.pi), 1.0)
    normalized = radii / equivalent
    angle = (np.arctan2(delta[:, 1], delta[:, 0]) + 2 * np.pi) % (2 * np.pi)
    bins = np.floor(angle / (2 * np.pi) * 72).astype(int) % 72
    profile = np.full(72, np.nan)
    for index in range(72):
        values = radii[bins == index]
        if len(values):
            profile[index] = values.max()
    good = np.flatnonzero(np.isfinite(profile))
    if len(good) < 2:
        profile[:] = equivalent
    else:
        extended_x = np.concatenate((good - 72, good, good + 72))
        extended_y = np.tile(profile[good], 3)
        profile = np.interp(np.arange(72), extended_x, extended_y)
    profile /= max(float(profile.mean()), 1e-6)
    spectrum = np.abs(np.fft.rfft(profile)) / max(len(profile), 1)
    result = {
        "rich_radial_mean": float(normalized.mean()),
        "rich_radial_std": float(normalized.std()),
        "rich_radial_q10": float(np.quantile(normalized, 0.10)),
        "rich_radial_q25": float(np.quantile(normalized, 0.25)),
        "rich_radial_median": float(np.quantile(normalized, 0.50)),
        "rich_radial_q75": float(np.quantile(normalized, 0.75)),
        "rich_radial_q90": float(np.quantile(normalized, 0.90)),
        "rich_radial_range": float(normalized.max() - normalized.min()),
        "rich_radial_skew": float(np.nan_to_num(skew(normalized), nan=0.0)),
        "rich_radial_kurtosis": float(np.nan_to_num(kurtosis(normalized), nan=0.0)),
    }
    for harmonic in range(1, 9):
        result[f"rich_radial_harmonic_{harmonic}"] = float(spectrum[harmonic])
    return result


def rich_features(mask: np.ndarray) -> dict[str, float]:
    mask = largest_component(mask)
    if not mask.any():
        raise RuntimeError("Empty topology mask")
    area = float(mask.sum())
    equivalent = max(math.sqrt(area / math.pi), 1.0)
    contour_list, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contour_list, key=cv2.contourArea)
    base_perimeter = max(perimeter(mask), 1.0)
    hull = cv2.convexHull(contour)
    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    deficit = (hull_mask > 0) & ~mask
    deficit_count, _, deficit_stats, _ = cv2.connectedComponentsWithStats(deficit.astype(np.uint8), 8)
    deficit_areas = deficit_stats[1:, cv2.CC_STAT_AREA] if deficit_count > 1 else np.asarray([])

    distance = ndi.distance_transform_edt(mask) / equivalent
    inside = distance[mask]
    skeleton = skeletonize(mask)
    neighbors = ndi.convolve(skeleton.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - skeleton
    holes = ndi.binary_fill_holes(mask) & ~mask
    hole_count, _, hole_stats, _ = cv2.connectedComponentsWithStats(holes.astype(np.uint8), 8)
    hole_areas = hole_stats[1:, cv2.CC_STAT_AREA] if hole_count > 1 else np.asarray([])

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).ravel()
    result = {f"rich_hu_{index + 1}": float(-np.sign(value) * np.log10(abs(value) + 1e-30)) for index, value in enumerate(hu)}
    result.update(radial_features(mask))
    result.update({
        "rich_distance_mean": float(inside.mean()),
        "rich_distance_std": float(inside.std()),
        "rich_distance_q10": float(np.quantile(inside, 0.10)),
        "rich_distance_q25": float(np.quantile(inside, 0.25)),
        "rich_distance_median": float(np.quantile(inside, 0.50)),
        "rich_distance_q75": float(np.quantile(inside, 0.75)),
        "rich_distance_q90": float(np.quantile(inside, 0.90)),
        "rich_distance_max": float(inside.max()),
        "rich_core_fraction_025": float((inside >= 0.25).mean()),
        "rich_core_fraction_050": float((inside >= 0.50).mean()),
        "rich_core_fraction_075": float((inside >= 0.75).mean()),
        "rich_convexity_deficit": float(deficit.sum() / max(hull_mask.sum(), 1)),
        "rich_largest_deficit_fraction": float(deficit_areas.max() / area) if len(deficit_areas) else 0.0,
        "rich_deficit_component_count": float(len(deficit_areas)),
        "rich_ellipse_iou": ellipse_iou(mask, contour),
        "rich_enclosing_circle_fill": float(area / max(math.pi * cv2.minEnclosingCircle(contour)[1] ** 2, 1.0)),
        "rich_skeleton_length_scaled": float(skeleton.sum() / equivalent),
        "rich_skeleton_endpoints": float(np.logical_and(skeleton, neighbors == 1).sum()),
        "rich_skeleton_branchpoints": float(np.logical_and(skeleton, neighbors >= 3).sum()),
        "rich_hole_count": float(len(hole_areas)),
        "rich_hole_area_fraction": float(hole_areas.sum() / area) if len(hole_areas) else 0.0,
    })
    for sigma in (1.0, 2.0, 4.0):
        smooth = ndi.gaussian_filter(mask.astype(np.float32), sigma=sigma) >= 0.5
        result[f"rich_perimeter_ratio_sigma_{int(sigma)}"] = float(perimeter(smooth) / base_perimeter)
    bitmap = aligned_bitmap(mask, 16)
    for index, value in enumerate(bitmap.ravel()):
        result[f"mask_pixel_{index:03d}"] = float(value)
    return result

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


def entropy(values: np.ndarray, bins: int = 32) -> float:
    if values.size == 0:
        return float("nan")
    histogram = np.histogram(values, bins=bins, range=(0.0, 1.0))[0].astype(float)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


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
        return {f"texture_glcm_{name}": float("nan") for name in ("contrast", "homogeneity", "energy", "entropy", "correlation")}
    probability = matrix / matrix.sum()
    i, j = np.indices(matrix.shape)
    mean_i = float((probability * i).sum())
    mean_j = float((probability * j).sum())
    std_i = math.sqrt(float((probability * (i - mean_i) ** 2).sum()))
    std_j = math.sqrt(float((probability * (j - mean_j) ** 2).sum()))
    positive = probability[probability > 0]
    return {
        "texture_glcm_contrast": float((probability * (i - j) ** 2).sum()),
        "texture_glcm_homogeneity": float((probability / (1.0 + (i - j) ** 2)).sum()),
        "texture_glcm_energy": float((probability ** 2).sum()),
        "texture_glcm_entropy": float(-(positive * np.log2(positive)).sum()),
        "texture_glcm_correlation": float((probability * (i - mean_i) * (j - mean_j)).sum() / (std_i * std_j)) if std_i * std_j > 0 else 0.0,
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
    for name, plane in zip(("r", "g", "b", "gray"), (*np.moveaxis(normalized, -1, 0), gray)):
        values = plane[mask]
        for statistic, value in zip(("mean", "std", "q10", "median", "q90"), (
            np.mean(values) if pixels else np.nan,
            np.std(values) if pixels else np.nan,
            np.quantile(values, 0.10) if pixels else np.nan,
            np.median(values) if pixels else np.nan,
            np.quantile(values, 0.90) if pixels else np.nan,
        )):
            features[f"texture_{name}_{statistic}"] = float(value)
        features[f"texture_{name}_entropy"] = entropy(values)

    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    for name, plane in (("gradient", gradient), ("laplacian", laplacian)):
        values = plane[mask]
        features[f"texture_{name}_mean"] = float(np.mean(values)) if pixels else np.nan
        features[f"texture_{name}_std"] = float(np.std(values)) if pixels else np.nan
        features[f"texture_{name}_q90"] = float(np.quantile(values, 0.90)) if pixels else np.nan

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
        histogram = np.bincount(lbp, minlength=256).astype(float)
        probabilities = histogram[histogram > 0] / histogram.sum()
        features["texture_lbp_entropy"] = float(-(probabilities * np.log2(probabilities)).sum())
        features["texture_lbp_mean"] = float(lbp.mean() / 255.0)
        features["texture_lbp_uniform_fraction"] = float(np.mean(np.isin(lbp, (0, 255, 1, 2, 4, 8, 16, 32, 64, 128))))
    else:
        features.update({"texture_lbp_entropy": np.nan, "texture_lbp_mean": np.nan, "texture_lbp_uniform_fraction": np.nan})
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
        features[f"texture_boundary_{name}_absolute_contrast"] = (
            float(abs(inside.mean() - outside.mean())) if inside.size and outside.size else np.nan
        )
    return features


def extract_candidates(image, cleaned_mask):
    if image.shape[:2] != cleaned_mask.shape:
        raise ValueError("Image and mask grids differ")
    whole = texture_features(image, np.ones(cleaned_mask.shape, dtype=bool))
    within = texture_features(image, cleaned_mask)
    morphology = {k: v for k, v in rich_features(cleaned_mask).items() if k.startswith("rich_")}
    if len(morphology) != 49 or len(within) != 42 or set(whole) != set(within):
        raise ValueError("Expected 49 morphology and 42 appearance candidates")
    return whole, within, morphology

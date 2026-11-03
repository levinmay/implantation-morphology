"""Fit a primary and satellite disk to a mask's largest component.

The Two-Circle Morphology Index (TCMI) uses the mask only:

    100 * neck_evidence * sqrt(asymmetric_fit_gain).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
from .cleanup import largest_component


MAXIMUM_NORMALIZED_SIDE = 128
MINIMUM_FOREGROUND_PIXELS = 20
PRIMARY_RADIUS_FACTORS = (0.92, 1.00, 1.08)
SATELLITE_RADIUS_RATIOS = (0.08, 0.12, 0.18, 0.26, 0.38, 0.55, 0.78, 1.00)
NORMALIZED_SEPARATIONS = (0.72, 0.84, 0.96)
ANGLE_OFFSETS_DEGREES = (-12.0, 0.0, 12.0)
RESIDUAL_OUTER_QUANTILE = 0.85


def normalized_mask(
    mask: np.ndarray, maximum_side: int = MAXIMUM_NORMALIZED_SIDE
) -> np.ndarray:
    """Crop the largest component and downsample it only when necessary."""
    value = largest_component(mask)
    if not value.any():
        return np.zeros((8, 8), dtype=bool)
    y, x = np.where(value)
    extent = max(x.max() - x.min() + 1, y.max() - y.min() + 1)
    pad = max(3, int(round(0.04 * extent)))
    y0 = max(0, int(y.min()) - pad)
    y1 = min(value.shape[0], int(y.max()) + pad + 1)
    x0 = max(0, int(x.min()) - pad)
    x1 = min(value.shape[1], int(x.max()) + pad + 1)
    crop = value[y0:y1, x0:x1]
    scale = min(1.0, maximum_side / max(crop.shape))
    if scale < 1.0:
        shape = (
            max(8, int(round(crop.shape[1] * scale))),
            max(8, int(round(crop.shape[0] * scale))),
        )
        crop = (
            cv2.resize(
                crop.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST
            )
            > 0
        )
    return crop


def _disk(shape: tuple[int, int], center: np.ndarray, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2


def _iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / max(union, 1))


def _convexity_deficit(mask: np.ndarray) -> float:
    """Return the fraction of the filled convex hull not occupied by a mask."""
    points_yx = np.argwhere(mask)
    if len(points_yx) < 3:
        return 0.0
    points_xy = points_yx[:, ::-1].astype(np.int32).reshape(-1, 1, 2)
    hull = cv2.convexHull(points_xy)
    hull_mask = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull_mask, hull, 1)
    hull_area = int(hull_mask.sum())
    return max(
        0.0,
        min(1.0, 1.0 - float(mask.sum()) / max(hull_area, 1)),
    )


def _best_radius(
    target: np.ndarray, center: np.ndarray, initial: float
) -> tuple[float, float]:
    best = (-1.0, max(initial, 1.0))
    for scale in np.linspace(0.65, 1.55, 37):
        radius = max(1.0, float(initial * scale))
        value = _iou(_disk(target.shape, center, radius), target)
        if (value, -radius) > (best[0], -best[1]):
            best = (value, radius)
    return best[1], best[0]


def _rotate_vector(vector: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    matrix = np.array(
        (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        )
    )
    return matrix @ vector


def _zero_metrics() -> dict[str, float]:
    return {
        "tcmi_score": 0.0,
        "two_circle_fit": 0.0,
        "asymmetric_fit_gain": 0.0,
        "normalized_center_separation": 0.0,
        "neck_evidence": 0.0,
        "satellite_radius_ratio": 0.0,
    }


def tcmi_metrics(mask: np.ndarray) -> dict[str, float]:
    """Compute TCMI and its mask-only audit components."""
    value = normalized_mask(mask)
    points = np.argwhere(value).astype(float)
    if len(points) < MINIMUM_FOREGROUND_PIXELS:
        return _zero_metrics()

    distance_map = cv2.distanceTransform(value.astype(np.uint8), cv2.DIST_L2, 5)
    maximum_index = int(np.argmax(distance_map))
    primary_center = np.array(
        np.unravel_index(maximum_index, distance_map.shape), dtype=float
    )
    primary_base_radius = float(distance_map.flat[maximum_index])
    if primary_base_radius < 2:
        return _zero_metrics()

    preliminary = _disk(value.shape, primary_center, 1.02 * primary_base_radius)
    residual_points = np.argwhere(value & ~preliminary).astype(float)
    directions: list[np.ndarray] = []
    if len(residual_points) >= 3:
        residual_distances = np.linalg.norm(residual_points - primary_center, axis=1)
        farthest = residual_points[
            residual_distances
            >= np.quantile(residual_distances, RESIDUAL_OUTER_QUANTILE)
        ]
        directions.append(farthest.mean(axis=0) - primary_center)

    covariance = np.cov(points, rowvar=False, bias=True)
    _, vectors = np.linalg.eigh(covariance)
    principal = vectors[:, -1]
    directions.extend((principal, -principal))

    unit_directions: list[np.ndarray] = []
    for direction in directions:
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-6:
            continue
        candidate = direction / norm
        if not any(
            float(np.dot(candidate, existing)) > 0.995
            for existing in unit_directions
        ):
            unit_directions.append(candidate)
    if not unit_directions:
        return _zero_metrics()

    one_center = points.mean(axis=0)
    one_radius, one_fit = _best_radius(
        value, one_center, math.sqrt(len(points) / math.pi)
    )
    primary_candidates: list[tuple[float, np.ndarray]] = []
    baseline_fit = one_fit
    baseline_geometry = (one_center, one_radius)
    for radius_factor in PRIMARY_RADIUS_FACTORS:
        primary_radius = primary_base_radius * radius_factor
        primary_mask = _disk(value.shape, primary_center, primary_radius)
        primary_fit = _iou(primary_mask, value)
        primary_candidates.append((primary_radius, primary_mask))
        if primary_fit > baseline_fit:
            baseline_fit = primary_fit
            baseline_geometry = (primary_center, primary_radius)

    best_fit = baseline_fit
    best_geometry = (
        baseline_geometry[0],
        baseline_geometry[1],
        np.zeros(2),
        0.0,
        baseline_fit,
    )
    for primary_radius, primary_mask in primary_candidates:
        for direction in unit_directions:
            for offset in ANGLE_OFFSETS_DEGREES:
                search_direction = _rotate_vector(direction, offset)
                for ratio in SATELLITE_RADIUS_RATIOS:
                    satellite_radius = primary_radius * ratio
                    for separation in NORMALIZED_SEPARATIONS:
                        center_distance = separation * (
                            primary_radius + satellite_radius
                        )
                        satellite_center = (
                            primary_center + search_direction * center_distance
                        )
                        satellite_mask = _disk(
                            value.shape, satellite_center, satellite_radius
                        )
                        union = primary_mask | satellite_mask
                        candidate_fit = _iou(union, value)
                        key = (candidate_fit, -ratio, -abs(separation - 0.84))
                        best_key = (
                            best_fit,
                            -(best_geometry[3] / max(best_geometry[1], 1e-6)),
                            -abs(
                                np.linalg.norm(best_geometry[2] - best_geometry[0])
                                / max(
                                    best_geometry[1] + best_geometry[3], 1e-6
                                )
                                - 0.84
                            )
                            if best_geometry[3] > 0
                            else -1.0,
                        )
                        if key > best_key:
                            best_fit = candidate_fit
                            best_geometry = (
                                primary_center.copy(),
                                primary_radius,
                                satellite_center.copy(),
                                satellite_radius,
                                candidate_fit,
                            )

    fitted_primary_center = best_geometry[0]
    fitted_primary_radius = best_geometry[1]
    satellite_center = best_geometry[2]
    satellite_radius = best_geometry[3]
    fit_gain = max(
        0.0,
        min(
            1.0,
            (best_fit - baseline_fit) / max(1.0 - baseline_fit, 1e-6),
        ),
    )
    if satellite_radius <= 0 or fit_gain <= 0:
        result = _zero_metrics()
        result["two_circle_fit"] = 100.0 * best_fit
        return result

    fitted_union = _disk(
        value.shape, fitted_primary_center, fitted_primary_radius
    ) | _disk(value.shape, satellite_center, satellite_radius)
    separation = min(
        1.0,
        float(np.linalg.norm(fitted_primary_center - satellite_center))
        / max(fitted_primary_radius + satellite_radius, 1e-6),
    )
    fitted_deficit = _convexity_deficit(fitted_union)
    neck_evidence = min(
        1.0,
        _convexity_deficit(value) / max(0.5 * fitted_deficit, 1e-6),
    )
    tcmi_score = 100.0 * neck_evidence * math.sqrt(fit_gain)
    return {
        "tcmi_score": float(tcmi_score),
        "two_circle_fit": 100.0 * best_fit,
        "asymmetric_fit_gain": fit_gain,
        "normalized_center_separation": separation,
        "neck_evidence": neck_evidence,
        "satellite_radius_ratio": satellite_radius
        / max(fitted_primary_radius, 1e-6),
    }


def tcmi_score(mask: np.ndarray) -> float:
    """Return only the primary TCMI value on its 0--100 scale."""
    return tcmi_metrics(mask)["tcmi_score"]

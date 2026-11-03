"""Six morphology and nine appearance descriptors."""

import json
import math
from pathlib import Path
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from .appearance import texture_features
from .cleanup import largest_component


def validate_feature_set(feature_set: dict) -> None:
    counts = {"rich_mask_morphology": 6, "whole_image_appearance": 9,
              "within_mask_appearance": 9, "combined": 15}
    for name, count in counts.items():
        values = feature_set.get(name, [])
        if len(values) != count or len(set(values)) != count:
            raise ValueError(f"Expected {count} unique descriptors in {name}")
    if feature_set["whole_image_appearance"] != feature_set["within_mask_appearance"]:
        raise ValueError("Whole-image and within-mask appearance names/order must match")
    union = set(feature_set["rich_mask_morphology"]) | set(feature_set["within_mask_appearance"])
    if set(feature_set["combined"]) != union:
        raise ValueError("Combined representation must be the exact descriptor union")


def morphology_features(mask: np.ndarray) -> dict[str, float]:
    mask = largest_component(np.asarray(mask, dtype=bool))
    if not mask.any():
        raise ValueError("The study morphology representation requires a nonempty mask")
    equivalent_radius = max(math.sqrt(float(mask.sum()) / math.pi), 1.0)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)
    hull = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(hull, cv2.convexHull(contour), 1)
    deficit = (hull > 0) & ~mask
    count, _, _, _ = cv2.connectedComponentsWithStats(deficit.astype(np.uint8), 8)
    center = np.argwhere(mask)[:, [1, 0]].astype(float).mean(axis=0)
    radii = np.linalg.norm(contour.reshape(-1, 2).astype(float) - center, axis=1)
    thickness = (ndi.distance_transform_edt(mask) / equivalent_radius)[mask]
    return {
        "rich_deficit_component_count": float(count - 1),
        "rich_skeleton_length_scaled": float(skeletonize(mask).sum() / equivalent_radius),
        "rich_radial_median": float(np.quantile(radii / equivalent_radius, 0.50)),
        "rich_core_fraction_075": float((thickness >= 0.75).mean()),
        "rich_distance_max": float(thickness.max()),
        "rich_distance_std": float(thickness.std()),
    }

def extract_features(image: np.ndarray, cleaned_mask: np.ndarray,
                     feature_set: dict, *, include_whole: bool = True) -> tuple[dict, dict, dict]:
    validate_feature_set(feature_set)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != cleaned_mask.shape:
        raise ValueError("Expected native RGB image and same-grid two-dimensional mask")
    whole = texture_features(image, np.ones(cleaned_mask.shape, dtype=bool)) if include_whole else {}
    within = texture_features(image, cleaned_mask.astype(bool))
    names = feature_set["within_mask_appearance"]
    return ({name: whole[name] for name in names} if include_whole else {},
            {name: within[name] for name in names}, morphology_features(cleaned_mask))


def default_feature_set() -> dict:
    return json.loads(Path(__file__).with_name("feature_set.json").read_text())

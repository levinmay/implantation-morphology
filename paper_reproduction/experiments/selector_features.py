"""Build proposal features and train the SAM mask selector."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


FULL_FEATURES = [
    "pred_iou",
    "stability",
    "log_area",
    "circ",
    "asp",
    "edge",
    "center_dist",
    "solidity",
    "num_cc",
    "lcr",
    "bbox_fill",
    "pred_iou_z",
    "log_area_z",
    "area_rank",
    "circ_rank",
    "fill_rank",
]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
        return default


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    reference = reference.astype(bool)
    denominator = int(prediction.sum()) + int(reference.sum())
    if denominator == 0:
        return 0.0
    return 2.0 * int(np.logical_and(prediction, reference).sum()) / denominator


def iou(prediction: np.ndarray, reference: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    reference = reference.astype(bool)
    union = int(np.logical_or(prediction, reference).sum())
    if union == 0:
        return 0.0
    return int(np.logical_and(prediction, reference).sum()) / union


def _bbox_aspect(box: list[float]) -> float:
    _, _, width, height = box
    if width <= 0 or height <= 0:
        return 0.0
    return float(min(width, height) / max(width, height))


def _perimeter(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return float(sum(cv2.arcLength(contour, True) for contour in contours))


def _circularity(mask: np.ndarray) -> float:
    area = float(mask.astype(bool).sum())
    perimeter = _perimeter(mask.astype(np.uint8) * 255)
    if area <= 0 or perimeter <= 0:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def _edge_flag(box: list[float], height: int, width: int, margin: int) -> float:
    x, y, box_width, box_height = box
    touches = (
        x < margin
        or y < margin
        or x + box_width > width - margin
        or y + box_height > height - margin
    )
    return float(touches)


def _center_distance(box: list[float], height: int, width: int) -> float:
    x, y, box_width, box_height = box
    center_x = x + 0.5 * box_width
    center_y = y + 0.5 * box_height
    scale = np.hypot(width / 2.0, height / 2.0)
    return float(np.hypot(center_x - width / 2.0, center_y - height / 2.0) / scale)


def _solidity(mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    return area / hull_area if area > 0 and hull_area > 0 else 0.0


def _component_stats(mask: np.ndarray) -> tuple[int, float]:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8))
    component_count = max(count - 1, 0)
    if component_count == 0:
        return 0, 0.0
    areas = sorted(
        (int((labels == label).sum()) for label in range(1, count)), reverse=True
    )
    return component_count, float(areas[0] / sum(areas))


def _bbox_fill(mask: np.ndarray, box: list[float]) -> float:
    _, _, width, height = box
    return float(mask.astype(bool).sum() / max(float(width * height), 1.0))


def _zscore(values: np.ndarray) -> np.ndarray:
    return (values - values.mean()) / (values.std() + 1e-8)


def _rank01(values: np.ndarray) -> np.ndarray:
    order = values.argsort()
    result = np.empty_like(order, dtype=float)
    result[order] = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
    return result


def proposal_feature_df(item: dict, edge_margin: int = 40) -> pd.DataFrame:
    """Construct the 16 proposal features and training targets for one image."""
    height, width = int(item["H"]), int(item["W"])
    reference = item["gt"].astype(bool)
    case_id = str(item.get("image_id", item.get("unit_id", "")))
    rows: list[dict] = []
    for proposal_index, annotation in enumerate(item["anns"]):
        mask = annotation["segmentation"].astype(bool)
        area = float(annotation.get("area", mask.sum()))
        box = annotation.get("bbox", [0, 0, width, height])
        component_count, largest_component_ratio = _component_stats(mask)
        rows.append(
            {
                "image_id": case_id,
                "prop_idx": proposal_index,
                "pred_iou": safe_float(annotation.get("predicted_iou", 0.0)),
                "stability": safe_float(annotation.get("stability_score", 0.0)),
                "log_area": float(np.log1p(max(area, 0.0))),
                "circ": _circularity(mask),
                "asp": _bbox_aspect(box),
                "edge": _edge_flag(box, height, width, edge_margin),
                "center_dist": _center_distance(box, height, width),
                "solidity": _solidity(mask),
                "num_cc": float(component_count),
                "lcr": largest_component_ratio,
                "bbox_fill": _bbox_fill(mask, box),
                "y_iou": iou(mask, reference),
                "y_dice": dice(mask, reference),
            }
        )
    if not rows:
        return pd.DataFrame()

    predicted_iou = np.asarray([row["pred_iou"] for row in rows], dtype=float)
    log_area = np.asarray([row["log_area"] for row in rows], dtype=float)
    circularity = np.asarray([row["circ"] for row in rows], dtype=float)
    fill = np.asarray([row["bbox_fill"] for row in rows], dtype=float)
    additions = {
        "pred_iou_z": _zscore(predicted_iou),
        "log_area_z": _zscore(log_area),
        "area_rank": _rank01(log_area),
        "circ_rank": _rank01(circularity),
        "fill_rank": _rank01(fill),
    }
    for index, row in enumerate(rows):
        for name, values in additions.items():
            row[name] = float(values[index])
    return pd.DataFrame(rows)


def build_ranker_df(items: list[dict]) -> pd.DataFrame:
    frames = [proposal_feature_df(item) for item in items]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_ranker_from_params(
    train_vis: list[dict],
    params: dict[str, float | int | None],
    feature_names: list[str] | None = None,
    seed: int = 0,
    train_df: pd.DataFrame | None = None,
) -> HistGradientBoostingRegressor | None:
    feature_names = feature_names or FULL_FEATURES
    frame = train_df if train_df is not None else build_ranker_df(train_vis)
    if frame.empty:
        return None
    model = HistGradientBoostingRegressor(
        random_state=seed,
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        max_leaf_nodes=params["max_leaf_nodes"],
        l2_regularization=params["l2_regularization"],
        min_samples_leaf=params["min_samples_leaf"],
    )
    model.fit(frame[feature_names].to_numpy(), frame["y_iou"].to_numpy())
    return model


def default_ranker_param_grid() -> list[dict[str, float | int | None]]:
    return [
        {
            "max_depth": depth,
            "learning_rate": rate,
            "max_leaf_nodes": leaves,
            "l2_regularization": regularization,
            "min_samples_leaf": minimum,
        }
        for depth in (None, 5)
        for rate in (0.03, 0.05, 0.1)
        for leaves in (15, 31, 63)
        for regularization in (0.0, 0.1, 1.0, 5.0)
        for minimum in (10, 20)
    ]


def smoke_ranker_param_grid() -> list[dict[str, float | int | None]]:
    return [
        {
            "max_depth": None,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.1,
            "min_samples_leaf": 10,
        }
    ]


def tune_ranker_params(
    train_vis: list[dict],
    val_vis: list[dict],
    feature_names: list[str] | None = None,
    seed: int = 0,
    param_grid: list[dict[str, float | int | None]] | None = None,
) -> dict[str, float | int | None]:
    feature_names = feature_names or FULL_FEATURES
    train_frame = build_ranker_df(train_vis)
    validation_frame = build_ranker_df(val_vis)
    best: dict[str, float | int | None] | None = None
    for parameters in param_grid or default_ranker_param_grid():
        model = train_ranker_from_params(
            train_vis, parameters, feature_names, seed, train_frame
        )
        if model is None:
            continue
        train_scored = train_frame.assign(
            ranker_score=model.predict(train_frame[feature_names].to_numpy())
        )
        validation_scored = validation_frame.assign(
            ranker_score=model.predict(validation_frame[feature_names].to_numpy())
        )
        train_indices = train_scored.groupby("image_id")["ranker_score"].idxmax()
        validation_indices = validation_scored.groupby("image_id")["ranker_score"].idxmax()
        candidate = dict(parameters)
        candidate["train_dice"] = float(train_scored.loc[train_indices, "y_dice"].mean())
        candidate["val_dice"] = float(
            validation_scored.loc[validation_indices, "y_dice"].mean()
        )
        candidate_key = (
            safe_float(candidate["val_dice"], -1.0),
            safe_float(candidate["train_dice"], -1.0),
            -safe_float(candidate["l2_regularization"]),
            safe_float(candidate["max_leaf_nodes"]),
            -safe_float(candidate["learning_rate"]),
            -safe_float(candidate["min_samples_leaf"]),
        )
        if best is None:
            best = candidate
        else:
            best_key = (
                safe_float(best["val_dice"], -1.0),
                safe_float(best["train_dice"], -1.0),
                -safe_float(best["l2_regularization"]),
                safe_float(best["max_leaf_nodes"]),
                -safe_float(best["learning_rate"]),
                -safe_float(best["min_samples_leaf"]),
            )
            if candidate_key > best_key:
                best = candidate
    if best is None:
        raise ValueError("No ranker parameter candidate could be evaluated")
    return best

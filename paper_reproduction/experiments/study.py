"""Shared input tables and mask scoring for the paper experiments."""

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from pipeline.staging import CLASSES as GROUPS, four_group as classify_group
from paper_reproduction.experiments.validation import validate_folds

EXPECTED_UNITS = 577


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_authority(release_dir: Path, split_dir: Path) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    manifest = read_csv(release_dir / "tables/embryo_mask_manifest.csv")
    assignments = read_csv(split_dir / "unit_fold_assignment.csv")
    if len(manifest) != EXPECTED_UNITS or len(assignments) != EXPECTED_UNITS:
        raise RuntimeError("Expected exact 577-row release and assignment tables")
    assignment_map = {row["unit_id"]: row for row in assignments}
    assignment_frame = pd.DataFrame(assignments)
    assignment_frame["evaluation_fold"] = pd.to_numeric(assignment_frame.evaluation_fold)
    validate_folds(assignment_frame, EXPECTED_UNITS, 123)
    if len(assignment_map) != EXPECTED_UNITS or set(assignment_map) != {row["unit_id"] for row in manifest}:
        raise RuntimeError("Release and split unit IDs do not join one-to-one")
    return manifest, assignment_map


def four_group(row: dict[str, str]) -> str:
    return classify_group(row["parent_label"], row.get("pre_substage", ""))


def dice(prediction: np.ndarray, reference: np.ndarray) -> float:
    denominator = int(prediction.sum()) + int(reference.sum())
    return 0.0 if denominator == 0 else 2.0 * int(np.logical_and(prediction, reference).sum()) / denominator


def boundary(mask: np.ndarray) -> np.ndarray:
    value = mask.astype(np.uint8)
    eroded = cv2.erode(value, np.ones((3, 3), np.uint8), iterations=1)
    return value.astype(bool) & ~eroded.astype(bool)


def enclosed_holes(mask: np.ndarray) -> np.ndarray:
    inverse = (~mask.astype(bool)).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(inverse, 8)
    border_labels = set(labels[0]) | set(labels[-1]) | set(labels[:, 0]) | set(labels[:, -1])
    return np.isin(labels, [label for label in range(1, count) if label not in border_labels])


def score_mask(item: dict[str, Any], method: str, mask: np.ndarray, score: float, proposal_index: int) -> dict[str, Any]:
    reference = item["gt"].astype(bool)
    prediction = mask.astype(bool)
    overlap = dice(prediction, reference)
    reference_area = int(reference.sum())
    prediction_area = int(prediction.sum())
    equivalent_diameter = max(1.0, 2.0 * math.sqrt(reference_area / math.pi))
    prediction_boundary = boundary(prediction)
    reference_boundary = boundary(reference)
    if prediction_boundary.any() and reference_boundary.any():
        distance_to_reference = cv2.distanceTransform(
            (~reference_boundary).astype(np.uint8), cv2.DIST_L2, 5
        )
        distance_to_prediction = cv2.distanceTransform(
            (~prediction_boundary).astype(np.uint8), cv2.DIST_L2, 5
        )
        prediction_distances = distance_to_reference[prediction_boundary]
        reference_distances = distance_to_prediction[reference_boundary]
        tolerance = max(1.0, round(0.01 * equivalent_diameter))
        normalized_surface_dice = float(
            (
                np.sum(prediction_distances <= tolerance)
                + np.sum(reference_distances <= tolerance)
            )
            / (len(prediction_distances) + len(reference_distances))
        )
        normalized_hd95 = float(
            max(
                np.quantile(prediction_distances, 0.95),
                np.quantile(reference_distances, 0.95),
            )
            / equivalent_diameter
        )
    else:
        normalized_surface_dice = 0.0
        normalized_hd95 = float("inf")
    component_count, _ = cv2.connectedComponents(
        prediction.astype(np.uint8), connectivity=8
    )
    topology_valid = component_count == 2 and not enclosed_holes(prediction).any()
    return {
        "unit_id": item["unit_id"],
        "embryo_id": item["embryo_id"],
        "experiment": item["experiment"],
        "dpf": item["dpf"],
        "four_group": item["four_group"],
        "component_id": item["component_id"],
        "evaluation_fold": item["evaluation_fold"],
        "method": method,
        "selection_score": score,
        "proposal_index": proposal_index,
        "proposal_count": len(item["anns"]),
        "dice": overlap,
        "nsd_1pct": normalized_surface_dice,
        "normalized_hd95": normalized_hd95,
        "relative_area_error": abs(prediction_area - reference_area) / reference_area,
        "single_component_hole_free": bool(topology_valid),
        "prediction_area_px": prediction_area,
        "reference_area_px": reference_area,
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frame = pd.DataFrame(rows)
    output = []
    for method, subset in frame.groupby("method"):
        output.append(
            {
                "method": method,
                "unit_count": len(subset),
                "overall_dice": float(subset["dice"].mean()),
                **{
                    f"{group}_dice": float(subset.loc[subset.four_group == group, "dice"].mean())
                    for group in GROUPS
                },
                "mean_relative_area_error": float(subset["relative_area_error"].mean()),
            }
        )
    return sorted(output, key=lambda row: row["method"])

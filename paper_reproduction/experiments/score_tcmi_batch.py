#!/usr/bin/env python3
"""Score reference and raw OOF masks for the 211 reviewed timing observations.

Eligibility and timing labels come from the review, not the mask score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from paper_reproduction.experiments.analyze_tcmi import TARGETS
from paper_reproduction.experiments.study import load_authority
from pipeline.io import sha256_path
from pipeline.tcmi import tcmi_score


def load_timing_labels(path: Path, known_units: set[str]) -> pd.DataFrame:
    labels = pd.read_csv(path, dtype={"unit_id": str})
    required = {"unit_id", *TARGETS}
    if not required.issubset(labels):
        raise ValueError(f"Missing timing columns: {sorted(required - set(labels))}")
    labels = labels[["unit_id", *TARGETS]].copy()
    if labels.isna().any().any() or labels.unit_id.duplicated().any():
        raise ValueError("Timing labels must have unique, nonmissing unit identifiers and targets")
    if len(labels) != 211 or not set(labels.unit_id).issubset(known_units):
        raise ValueError("Expected 211 eligible units present in the governed manifest")
    for target, positives in zip(TARGETS, (62, 60)):
        if set(labels[target]) != {0, 1} or int(labels[target].sum()) != positives:
            raise ValueError(f"{target} must be binary with exactly {positives} positives")
        labels[target] = labels[target].astype(int)
    if (labels.exact_hatching_day_target > labels.hatching_window_target).any():
        raise ValueError("Exact-day positives must also be hatching-window positives")
    return labels


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def score_pairs(manifest: list[dict], assignments: dict, labels: pd.DataFrame,
                release_dir: Path, nnunet_dir: Path) -> pd.DataFrame:
    """Score masks independently of labels, preserving manifest observation order."""
    eligible_units = set(labels.unit_id)
    rows = []
    cache: dict[tuple, float] = {}
    for row in manifest:
        unit = row["unit_id"]
        if unit not in eligible_units:
            continue
        split = assignments[unit]
        reference = load_mask(release_dir / row["mask_path"])
        raw = load_mask(nnunet_dir / "folds" / f"fold_{split['evaluation_fold']}"
                        / "raw_predictions" / f"{unit}.png")
        if reference.shape != raw.shape:
            raise ValueError(f"Reference/raw prediction grid mismatch for {unit}")
        if not reference.any():
            raise ValueError(f"The study requires a nonempty reference mask for {unit}")
        for source, mask in (("reference", reference), ("nnunet_raw", raw)):
            key = (mask.shape, hashlib.sha256(mask.tobytes()).digest())
            if key not in cache:
                cache[key] = tcmi_score(mask)
            rows.append({"unit_id": unit, "component_id": split["component_id"],
                         "source": source, "tcmi_score": cache[key]})
    scores = pd.DataFrame(rows).merge(labels, on="unit_id", how="left",
                                      validate="many_to_one", sort=False)
    if len(scores) != 422 or scores[list(TARGETS)].isna().any().any():
        raise ValueError("Incomplete paired TCMI score coverage")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release-dir", "split-dir", "nnunet-dir", "timing-labels", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Choose a new TCMI output directory; existing runs are preserved")
    manifest, assignments = load_authority(args.release_dir, args.split_dir)
    labels = load_timing_labels(args.timing_labels, set(assignments))
    scores = score_pairs(manifest, assignments, labels, args.release_dir, args.nnunet_dir)
    private_dir = args.output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=False)
    score_path = private_dir / "TCMI_SCORES.csv"
    scores.to_csv(score_path, index=False)
    provenance = {
        "status": "PASS", "eligible_units": 211, "score_rows": 422,
        "scores_use_raw_predictions": True, "timing_labels_are_metric_inputs": False,
        "input_sha256": {
            "manifest": sha256_path(args.release_dir / "tables/embryo_mask_manifest.csv"),
            "fold_assignment": sha256_path(args.split_dir / "unit_fold_assignment.csv"),
            "timing_labels": sha256_path(args.timing_labels),
        },
        "scores_sha256": sha256_path(score_path),
    }
    (private_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("PASS: 422 paired TCMI scores written; row-level outputs must remain private")


if __name__ == "__main__":
    main()

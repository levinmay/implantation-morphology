#!/usr/bin/env python3
"""Run the paper's fixed developmental-state representation comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from paper_reproduction.experiments.stage_classifier import (
    CLASSES,
    EXPECTED_COUNTS,
    class_metrics,
    four_group,
    metric_intervals,
    metrics,
    nested_oof,
)
from paper_reproduction.experiments.validation import validate_feature_set, validate_folds, validate_outer_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-feature-dir", type=Path, required=True,
                        help="Contains outer_fold_0..4/private feature caches")
    parser.add_argument("--feature-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    first = args.outer_feature_dir / "outer_fold_0" / "private"
    appearance_cache = pd.read_csv(first / "APPEARANCE_FEATURES.csv")
    morphology_cache = pd.read_csv(first / "MORPHOLOGY_FEATURES.csv")
    feature_set = json.loads(args.feature_set.read_text())
    validate_feature_set(feature_set)
    combined = [str(feature) for feature in feature_set["combined"]]
    morphology = [str(feature) for feature in feature_set["rich_mask_morphology"]]
    whole_appearance = [str(feature) for feature in feature_set["whole_image_appearance"]]
    within_appearance = [str(feature) for feature in feature_set["within_mask_appearance"]]
    # Use the six morphology features listed in the paper.
    morphology_cache = morphology_cache[["unit_id", *morphology]].copy()
    mask_units = appearance_cache.loc[appearance_cache.mask_source == "nnunet_250_topology", "unit_id"]
    if morphology_cache.unit_id.duplicated().any() or set(morphology_cache.unit_id) != set(mask_units):
        raise ValueError("Morphology cache must cover exactly the masked units")
    if len(morphology) != 6 or len(whole_appearance) != 9 or len(within_appearance) != 9:
        raise RuntimeError(
            "Expected 6 morphology, 9 whole-image appearance, and 9 "
            "within-mask appearance features"
        )

    def prepare(source: str, include_morphology: bool) -> pd.DataFrame:
        frame = appearance_cache.loc[appearance_cache.mask_source == source].copy()
        if include_morphology:
            frame = frame.merge(morphology_cache, on="unit_id", how="left", validate="one_to_one")
        frame["four_group"] = frame.apply(
            lambda row: four_group(row.parent_label, row.pre_substage), axis=1
        )
        frame["target"] = frame.four_group.map(CLASSES.index)
        if len(frame) != 577:
            raise RuntimeError(f"Unexpected {source} row count: {len(frame)}")
        validate_folds(frame, 577, 123)
        if frame.four_group.value_counts().to_dict() != EXPECTED_COUNTS:
            raise ValueError("Unexpected developmental-group counts")
        return frame

    whole = prepare("whole_crop", False)
    outer_frames = {}
    for fold in range(5):
        cache_dir = args.outer_feature_dir / f"outer_fold_{fold}" / "private"
        provenance = json.loads((cache_dir / "PROVENANCE.json").read_text())
        if (provenance["producer_outer_fold"] != fold
                or provenance["outer_test_excluded_from_fitting_and_planning"] is not True):
            raise ValueError("Missing upstream fold-isolation verification")
        appearance_cache = pd.read_csv(cache_dir / "APPEARANCE_FEATURES.csv")
        morphology_cache = pd.read_csv(cache_dir / "MORPHOLOGY_FEATURES.csv")
        if set(morphology_cache.producer_outer_fold) != {fold}:
            raise ValueError("Morphology cache has a different mask producer")
        morphology_cache = morphology_cache[["unit_id", *morphology]]
        outer_frames[fold] = prepare("nnunet_250_topology", True)
    validate_outer_frames(outer_frames)
    masked = outer_frames[0]
    identity = ["unit_id", "component_id", "evaluation_fold", "target", "four_group"]
    if not whole[identity].sort_values("unit_id").reset_index(drop=True).equals(
        masked[identity].sort_values("unit_id").reset_index(drop=True)
    ):
        raise ValueError("Appearance and mask representations are not paired")
    specifications = {
        "whole_image_compact_appearance": (whole, whole_appearance),
        "rich_mask_compact_morphology": (masked, morphology),
        "within_mask_compact_appearance": (masked, within_appearance),
        "compact_morphology_plus_appearance": (masked, combined),
    }

    aggregate_dir = args.output_dir / "aggregate"
    private_dir = args.output_dir / "private"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    for method, (frame, features) in specifications.items():
        predicted, selected = nested_oof(
            frame, features,
            outer_frames=None if method == "whole_image_compact_appearance" else outer_frames,
            model_dir=private_dir / "models" / method,
        )
        predicted.to_csv(private_dir / f"predictions_{method}.csv", index=False)
        values = metrics(predicted)
        summary_rows.append({
            "method": method,
            "feature_count": len(features),
            **values,
            **metric_intervals(predicted),
        })
        selected_rows.extend({"method": method, **row} for row in selected)
        class_rows.extend({"method": method, **row} for row in class_metrics(predicted))
        print(json.dumps({"method": method, "features": len(features), **values}), flush=True)

    pd.DataFrame(summary_rows).to_csv(
        aggregate_dir / "MATCHED_COMPACT_ABLATION.csv", index=False
    )
    pd.DataFrame(selected_rows).to_csv(
        aggregate_dir / "SELECTED_MODELS.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        aggregate_dir / "PER_CLASS_METRICS.csv", index=False
    )
    (aggregate_dir / "FEATURE_SET.json").write_text(json.dumps({
        "whole_image_appearance": whole_appearance,
        "rich_mask_morphology": morphology,
        "within_mask_appearance": within_appearance,
        "combined": combined,
    }, indent=2) + "\n")
    validation = {
        "status": "PASS",
        "posthoc_exploratory": True,
        "confirmatory": False,
        "units": 577,
        "morphology_features": len(morphology),
        "whole_image_appearance_features": len(whole_appearance),
        "within_mask_appearance_features": len(within_appearance),
        "combined_features": len(combined),
        "fold_protocol": "test k; validation (k+1) mod 5; remaining folds train",
        "upstream_mask_protocol": "one segmenter per outer fold for all staging rows",
        "upstream_training_masks": "in-sample; outer-test masks remain out-of-fold",
        "aggregate_contains_row_level_data": False,
        "private_row_level_directory": "private",
    }
    (aggregate_dir / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(validation, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

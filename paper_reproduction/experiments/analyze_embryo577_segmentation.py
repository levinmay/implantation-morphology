#!/usr/bin/env python3
"""Create the paper's aggregate segmentation and topology tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from paper_reproduction.experiments.validation import validate_folds


METHODS = (
    "sam_largest",
    "sam_stability",
    "sam_predicted_iou",
    "learned_iou",
    "nnunet_250",
)
GROUPS = (
    "early_pre_implantation_hatching",
    "floating_apposition_adhesion",
    "implantation_or_later",
    "degeneration",
)
# Map internal group names to published CSV columns.
GROUP_COLUMNS = {
    group: ("early_preimplantation_or_hatching_dice"
            if group == "early_pre_implantation_hatching" else f"{group}_dice")
    for group in GROUPS
}
COMPARISONS = (
    ("learned_iou", "sam_predicted_iou"),
    ("nnunet_250", "learned_iou"),
    ("nnunet_250", "sam_predicted_iou"),
)
SEED = 20260827
EXPECTED_UNITS = 577


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-metrics", type=Path, required=True)
    parser.add_argument("--nnunet-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sam = pd.read_csv(args.sam_metrics)
    nnunet = pd.read_csv(args.nnunet_metrics)
    frames = {
        method: sam.loc[sam.method == method].sort_values("unit_id").reset_index(drop=True)
        for method in METHODS if method != "nnunet_250"
    }
    frames["nnunet_250"] = (
        nnunet.loc[nnunet.method == "nnunet_250"]
        .sort_values("unit_id")
        .reset_index(drop=True)
    )
    identity_columns = ["unit_id", "component_id", "evaluation_fold", "four_group"]
    identity = frames[METHODS[0]][identity_columns]
    validate_folds(identity, 577, 123)
    for method, frame in frames.items():
        if len(frame) != EXPECTED_UNITS or frame.unit_id.nunique() != EXPECTED_UNITS:
            raise RuntimeError(f"Unexpected row count for {method}: {len(frame)}")
        if not identity.equals(frame[identity_columns]):
            raise RuntimeError(f"Metrics are not paired for {method}")

    summary_rows = []
    for method in METHODS:
        frame = frames[method]
        row = {
            "method": method,
            "units": len(frame),
            "overall_dice": float(frame.dice.mean()),
            "mean_relative_area_error": float(frame.relative_area_error.mean()),
        }
        for group in GROUPS:
            row[GROUP_COLUMNS[group]] = float(
                frame.loc[frame.four_group == group, "dice"].mean()
            )
        summary_rows.append(row)

    group_ids = sorted(identity.component_id.unique())
    group_index = {group: index for index, group in enumerate(group_ids)}
    row_group = identity.component_id.map(group_index).to_numpy(int)
    generator = np.random.default_rng(SEED)
    draws = {comparison: [] for comparison in COMPARISONS}
    for _ in range(args.bootstrap_repetitions):
        counts = generator.multinomial(
            len(group_ids), np.full(len(group_ids), 1.0 / len(group_ids))
        )
        weights = counts[row_group].astype(float)
        for method_a, method_b in COMPARISONS:
            value_a = np.average(frames[method_a].dice, weights=weights)
            value_b = np.average(frames[method_b].dice, weights=weights)
            draws[(method_a, method_b)].append(value_a - value_b)

    interval_rows = []
    for method_a, method_b in COMPARISONS:
        values = draws[(method_a, method_b)]
        value_a = float(frames[method_a].dice.mean())
        value_b = float(frames[method_b].dice.mean())
        low, high = np.quantile(values, (0.025, 0.975))
        interval_rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "metric": "dice",
            "method_a_value": value_a,
            "method_b_value": value_b,
            "difference_a_minus_b": value_a - value_b,
            "ci_low": float(low),
            "ci_high": float(high),
            "higher_is_better": True,
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "resampling_unit": "group",
        })

    topology_rows = []
    for method, label in (
        ("nnunet_250", "raw_nnunet"),
        ("nnunet_250_topology", "topology_standardized"),
    ):
        frame = nnunet.loc[nnunet.method == method]
        if len(frame) != EXPECTED_UNITS:
            raise RuntimeError(f"Unexpected topology row count for {method}")
        ordered = frame.sort_values("unit_id").reset_index(drop=True)
        if not identity.equals(ordered[identity_columns]):
            raise ValueError(f"Cleanup metrics are not paired for {method}")
        topology_rows.append({
            "output": label,
            "units": len(frame),
            "dice": float(frame.dice.mean()),
            "nsd_1pct": float(frame.nsd_1pct.mean()),
            "normalized_hd95": float(frame.normalized_hd95.mean()),
            "relative_area_error": float(frame.relative_area_error.mean()),
            "single_component_hole_free_rate": float(
                frame.single_component_hole_free.astype(str).str.lower().eq("true").mean()
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.output_dir / "segmentation_results.csv", index=False
    )
    pd.DataFrame(interval_rows).to_csv(
        args.output_dir / "segmentation_paired_bootstrap.csv", index=False
    )
    pd.DataFrame(topology_rows).to_csv(
        args.output_dir / "topology_standardization.csv", index=False
    )
    validation = {
        "status": "PASS",
        "units": EXPECTED_UNITS,
        "methods": list(METHODS),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "row_level_output_written": False,
    }
    (args.output_dir / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Aggregate TCMI timing associations from private row-level scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


TARGETS = ("hatching_window_target", "exact_hatching_day_target")
# Original seed: 20260827 + metric index 2.
SEED = 20260829


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


def bootstrap_auc(
    frame: pd.DataFrame, target: str, repetitions: int, seed: int
) -> tuple[float, float]:
    # Keep the original component and row order so bootstrap draws match.
    grouped = {component: rows for component, rows in frame.groupby("component_id")}
    groups = sorted(grouped)
    targets = {group: grouped[group][target].to_numpy() for group in groups}
    scores = {group: grouped[group].tcmi_score.to_numpy() for group in groups}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled = rng.choice(groups, len(groups), replace=True)
        sample_targets = np.concatenate([targets[group] for group in sampled])
        if np.unique(sample_targets).size == 2:
            sample_scores = np.concatenate([scores[group] for group in sampled])
            values.append(roc_auc_score(sample_targets, sample_scores))
    if not values:
        raise ValueError("No bootstrap resample contained both timing classes")
    return tuple(float(value) for value in np.quantile(values, (0.025, 0.975)))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.scores)
    required = {"unit_id", "component_id", "source", "tcmi_score", *TARGETS}
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(f"Missing columns: {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing timing score or identifier")
    if set(frame.source) != {"reference", "nnunet_raw"}:
        raise ValueError("Expected reference and raw out-of-fold nnU-Net scores")
    if not np.isfinite(frame.tcmi_score).all() or not frame.tcmi_score.between(0, 100).all():
        raise ValueError("TCMI must be finite and between 0 and 100")
    identity = ["unit_id", "component_id", *TARGETS]
    paired = [group.sort_values("unit_id").reset_index(drop=True)
              for _, group in frame.groupby("source")]
    if not paired[0][identity].equals(paired[1][identity]):
        raise ValueError("Mask-source timing observations are not paired")
    rows = []
    for source, subset in frame.groupby("source"):
        if len(subset) != 211 or subset.unit_id.nunique() != 211:
            raise RuntimeError(f"Expected 211 eligible observations for {source}")
        for target in TARGETS:
            if set(subset[target]) != {0, 1}:
                raise ValueError("Timing targets must contain binary 0/1 labels")
            low, high = bootstrap_auc(
                subset,
                target,
                args.bootstrap_repetitions,
                SEED,
            )
            scores = subset.tcmi_score.to_numpy(float)
            # Summaries use positives; AUROC uses all 211 observations.
            values = subset.loc[subset[target] == 1, "tcmi_score"].to_numpy(float)
            rows.append({
                "source": source,
                "target": target,
                "n": len(subset),
                "positive_n": int(subset[target].sum()),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "zero_fraction": float(np.mean(values == 0)),
                "above_50_fraction": float(np.mean(values > 50)),
                "above_70_fraction": float(np.mean(values > 70)),
                "auroc": float(roc_auc_score(subset[target], scores)),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_reps": args.bootstrap_repetitions,
            })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "tcmi_aggregate.csv", index=False)
    validation = {
        "status": "PASS_EXPLORATORY",
        "eligible_units_per_source": 211,
        "mask_sources": sorted(frame.source.unique().tolist()),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": SEED,
        "descriptive_summary_population": "target-positive observations",
        "labels_used_only_for_aggregate_validation": True,
        "row_level_outputs_private": True,
    }
    (args.output_dir / "tcmi_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

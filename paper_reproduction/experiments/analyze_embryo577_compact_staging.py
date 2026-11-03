#!/usr/bin/env python3
"""Compute component-bootstrap intervals and calibration from OOF predictions."""

from __future__ import annotations

import argparse
from pipeline.io import sha256_path as sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pipeline.staging import CLASSES, SEED
from paper_reproduction.experiments.validation import validate_folds
from sklearn.metrics import log_loss
from paper_reproduction.experiments.stage_classifier import (
    probability_matrix as probability, metrics as classification_metrics,
)


METHODS = (
    "whole_image_compact_appearance",
    "rich_mask_compact_morphology",
    "within_mask_compact_appearance",
    "compact_morphology_plus_appearance",
)
PROPOSED = "compact_morphology_plus_appearance"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def metrics(frame: pd.DataFrame, weights: np.ndarray | None = None) -> dict[str, float]:
    return classification_metrics(frame, weights, include_losses=False)


def load_predictions(directory: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames: dict[str, pd.DataFrame] = {}
    hashes: dict[str, str] = {}
    identity = ["unit_id", "component_id", "evaluation_fold", "target", "four_group"]
    reference: pd.DataFrame | None = None
    for method in METHODS:
        path = directory / f"predictions_{method}.csv"
        frame = pd.read_csv(path).sort_values("unit_id").reset_index(drop=True)
        validate_folds(frame, 577, 123)
        if len(frame) != 577 or frame.unit_id.nunique() != 577:
            raise RuntimeError(f"Unexpected row count for {method}: {len(frame)}")
        if frame.component_id.nunique() != 123:
            raise RuntimeError(f"Unexpected component count for {method}")
        if reference is None:
            reference = frame[identity].copy()
        elif not reference.equals(frame[identity]):
            raise RuntimeError(f"Prediction identity is not paired for {method}")
        frames[method] = frame
        hashes[str(path)] = sha256(path)
    return frames, hashes


def paired_bootstrap(
    frames: dict[str, pd.DataFrame], repetitions: int
) -> pd.DataFrame:
    proposed = frames[PROPOSED]
    groups = sorted(proposed.component_id.unique())
    group_index = {group: index for index, group in enumerate(groups)}
    row_group = proposed.component_id.map(group_index).to_numpy(int)
    generator = np.random.default_rng(SEED)
    metric_names = (
        "macro_auroc",
        "balanced_accuracy",
        "macro_f1",
        *(f"{label}_auroc" for label in CLASSES),
    )
    observed = {method: metrics(frame) for method, frame in frames.items()}
    draws = {
        baseline: {metric: [] for metric in metric_names}
        for baseline in METHODS if baseline != PROPOSED
    }
    successful = 0
    for _ in range(repetitions):
        counts = generator.multinomial(
            len(groups), np.full(len(groups), 1.0 / len(groups))
        )
        weights = counts[row_group].astype(float)
        try:
            sampled = {method: metrics(frame, weights) for method, frame in frames.items()}
        except ValueError:
            continue
        successful += 1
        for baseline in draws:
            for metric in metric_names:
                draws[baseline][metric].append(
                    sampled[PROPOSED][metric] - sampled[baseline][metric]
                )
    rows = []
    for baseline, metric_draws in draws.items():
        for metric, values in metric_draws.items():
            low, high = np.quantile(values, (0.025, 0.975))
            rows.append({
                "proposed_method": PROPOSED,
                "baseline_method": baseline,
                "metric": metric,
                "proposed_value": observed[PROPOSED][metric],
                "baseline_value": observed[baseline][metric],
                "difference": observed[PROPOSED][metric] - observed[baseline][metric],
                "ci_low": float(low),
                "ci_high": float(high),
                "successful_repetitions": successful,
                "requested_repetitions": repetitions,
                "resampling_unit": "group",
            })
    return pd.DataFrame(rows)


def calibration_tables(
    frame: pd.DataFrame, bin_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    target = frame.target.to_numpy(int)
    scores = probability(frame)
    bins_rows = []
    summary_rows = []
    for index, label in enumerate(CLASSES):
        binary = (target == index).astype(float)
        class_scores = scores[:, index]
        ordered = np.argsort(class_scores, kind="stable")
        ece = 0.0
        maximum_gap = 0.0
        for bin_index, members in enumerate(np.array_split(ordered, bin_count), start=1):
            mean_probability = float(class_scores[members].mean())
            observed_fraction = float(binary[members].mean())
            gap = abs(mean_probability - observed_fraction)
            ece += len(members) / len(frame) * gap
            maximum_gap = max(maximum_gap, gap)
            bins_rows.append({
                "class": label,
                "bin": bin_index,
                "n": len(members),
                "minimum_probability": float(class_scores[members].min()),
                "maximum_probability": float(class_scores[members].max()),
                "mean_probability": mean_probability,
                "observed_fraction": observed_fraction,
                "absolute_gap": gap,
            })
        summary_rows.append({
            "scope": label,
            "n": len(frame),
            "prevalence": float(binary.mean()),
            "mean_probability": float(class_scores.mean()),
            "brier": float(np.mean((class_scores - binary) ** 2)),
            "ece_equal_frequency_10_bin": float(ece),
            "maximum_bin_gap": float(maximum_gap),
            "multiclass_brier": np.nan,
            "log_loss": np.nan,
        })
    class_ece = [row["ece_equal_frequency_10_bin"] for row in summary_rows]
    one_hot = np.eye(len(CLASSES))[target]
    summary_rows.append({
        "scope": "overall",
        "n": len(frame),
        "prevalence": np.nan,
        "mean_probability": np.nan,
        "brier": np.nan,
        "ece_equal_frequency_10_bin": float(np.mean(class_ece)),
        "maximum_bin_gap": float(max(row["maximum_bin_gap"] for row in summary_rows)),
        "multiclass_brier": float(np.mean(np.sum((scores - one_hot) ** 2, axis=1))),
        "log_loss": float(log_loss(target, scores, labels=np.arange(len(CLASSES)))),
    })
    return pd.DataFrame(bins_rows), pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    frames, hashes = load_predictions(args.predictions_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contrasts = paired_bootstrap(frames, args.bootstrap_repetitions)
    bins, calibration = calibration_tables(frames[PROPOSED], args.calibration_bins)
    contrasts.to_csv(args.output_dir / "PAIRED_COMPONENT_BOOTSTRAP.csv", index=False)
    bins.to_csv(args.output_dir / "CALIBRATION_BINS.csv", index=False)
    calibration.to_csv(args.output_dir / "CALIBRATION_SUMMARY.csv", index=False)
    validation = {
        "status": "PASS",
        "units": 577,
        "components": 123,
        "classes": list(CLASSES),
        "methods": list(METHODS),
        "proposed_method": PROPOSED,
        "bootstrap_repetitions_requested": args.bootstrap_repetitions,
        "bootstrap_repetitions_successful": int(contrasts.successful_repetitions.min()),
        "bootstrap_unit": "group",
        "bootstrap_seed": SEED,
        "calibration_bins": args.calibration_bins,
        "calibration_binning": "equal-frequency within each one-vs-rest class",
        "input_sha256": hashes,
        "row_level_output_written": False,
        "evaluation_scope": "internal grouped five-fold out-of-fold predictions",
    }
    (args.output_dir / "VALIDATION.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n"
    )
    print(contrasts.to_csv(index=False), flush=True)
    print(calibration.to_csv(index=False), flush=True)
    print(json.dumps(validation, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

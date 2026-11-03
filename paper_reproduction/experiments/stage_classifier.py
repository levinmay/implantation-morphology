"""Shared four-group developmental-state classification utilities."""

from __future__ import annotations

from typing import Any, Iterable
from pathlib import Path
import json
import sklearn

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from paper_reproduction.experiments.validation import validate_folds, validate_outer_frames
from pipeline.staging import CLASSES, four_group, candidates, aligned_probability, save_model
from pipeline.io import write_checksums


EXPECTED_COUNTS = {
    "early_pre_implantation_hatching": 87,
    "floating_apposition_adhesion": 129,
    "implantation_or_later": 207,
    "degeneration": 154,
}
SEED = 20260827
BOOTSTRAP_REPETITIONS = 5000


def nested_oof(
    frame: pd.DataFrame, features: list[str],
    *, outer_frames: dict[int, pd.DataFrame] | None = None,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    outputs: list[pd.DataFrame] = []
    if outer_frames is not None:
        validate_outer_frames(outer_frames)
    validate_folds(frame)
    if len(features) != len(set(features)) or not set(features).issubset(frame):
        raise ValueError("Missing or duplicate feature names")
    if model_dir is not None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=False)
    selected_rows: list[dict[str, Any]] = []
    for fold in range(5):
        if outer_frames is not None:
            frame = outer_frames[fold]
            if not set(features).issubset(frame):
                raise ValueError("Missing outer-fold features")
        validation_fold = (fold + 1) % 5
        train = frame.loc[~frame.evaluation_fold.isin((fold, validation_fold))]
        validation = frame.loc[frame.evaluation_fold == validation_fold]
        test = frame.loc[frame.evaluation_fold == fold]
        best: tuple[float, str, Pipeline] | None = None
        for name, estimator in candidates(features, fold):
            estimator.fit(train[features], train.target)
            probability = aligned_probability(estimator, validation[features])
            score = float(roc_auc_score(
                validation.target,
                probability,
                multi_class="ovr",
                average="macro",
                labels=np.arange(len(CLASSES)),
            ))
            key = (score, name)
            if best is None or key > (best[0], best[1]):
                best = (score, name, estimator)
        if best is None:
            raise RuntimeError("No candidate estimator selected")
        _, name, estimator = best
        combined = pd.concat((train, validation), ignore_index=True)
        estimator.fit(combined[features], combined.target)
        probability = aligned_probability(estimator, test[features])
        if model_dir is not None:
            save_model({
                "schema_version": 1, "features": list(features), "classes": list(CLASSES),
                "sklearn_version": sklearn.__version__, "model_name": name,
                "seed": SEED + fold, "estimator": estimator, "outer_fold": fold,
                "origin": "outer_fold_training_and_validation_refit",
                "producer_outer_fold": fold if outer_frames is not None else None,
            }, model_dir / f"outer_fold_{fold}.joblib")
        predicted = test[
            ["unit_id", "component_id", "evaluation_fold", "target", "four_group"]
        ].copy()
        for index, label in enumerate(CLASSES):
            predicted[f"prob_{label}"] = probability[:, index]
        outputs.append(predicted)
        selected_rows.append({
            "fold": fold,
            "validation_fold": validation_fold,
            "selected_model": name,
            "validation_macro_auroc": best[0],
            "train_units": len(train),
            "validation_units": len(validation),
            "test_units": len(test),
            "feature_count": len(features),
            **({"producer_outer_fold": fold} if outer_frames is not None else {}),
        })
    result = pd.concat(outputs, ignore_index=True).sort_values("unit_id")
    if len(result) != len(frame) or result.unit_id.nunique() != len(frame):
        raise RuntimeError("Incomplete or duplicate out-of-fold predictions")
    if model_dir is not None:
        with (model_dir / "COMPLETE.json").open("x") as handle:
            json.dump({"outer_folds": 5, "features": list(features),
                       "status": "complete"}, handle, indent=2)
        write_checksums(model_dir)
    return result, selected_rows


def probability_matrix(frame: pd.DataFrame) -> np.ndarray:
    return frame[[f"prob_{label}" for label in CLASSES]].to_numpy(float)


def metrics(
    frame: pd.DataFrame, weights: np.ndarray | None = None, *, include_losses: bool = True
) -> dict[str, float]:
    target = frame.target.to_numpy(int)
    probability = probability_matrix(frame)
    prediction = probability.argmax(axis=1)
    result = {
        "macro_auroc": float(roc_auc_score(
            target,
            probability,
            multi_class="ovr",
            average="macro",
            labels=np.arange(len(CLASSES)),
            sample_weight=weights,
        )),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction, sample_weight=weights)
        ),
        "macro_f1": float(
            f1_score(target, prediction, average="macro", sample_weight=weights)
        ),
    }
    if include_losses:
        result.update({
            "log_loss": float(log_loss(
                target,
                probability,
                labels=np.arange(len(CLASSES)),
                sample_weight=weights,
            )),
            "multiclass_brier": float(np.average(
                np.sum((probability - np.eye(len(CLASSES))[target]) ** 2, axis=1),
                weights=weights,
            )),
        })
    for index, label in enumerate(CLASSES):
        binary = (target == index).astype(int)
        result[f"{label}_auroc"] = float(
            roc_auc_score(binary, probability[:, index], sample_weight=weights)
        )
    return result


def grouped_weights(
    frame: pd.DataFrame, repetitions: int, seed: int
) -> Iterable[np.ndarray]:
    groups = sorted(frame.component_id.unique())
    indexes = {group: index for index, group in enumerate(groups)}
    row_groups = frame.component_id.map(indexes).to_numpy(int)
    generator = np.random.default_rng(seed)
    for _ in range(repetitions):
        counts = generator.multinomial(
            len(groups), np.full(len(groups), 1.0 / len(groups))
        )
        yield counts[row_groups].astype(float)


def metric_intervals(frame: pd.DataFrame) -> dict[str, float]:
    names = ("macro_auroc", "balanced_accuracy", "macro_f1")
    values = {name: [] for name in names}
    for weights in grouped_weights(frame, BOOTSTRAP_REPETITIONS, SEED):
        try:
            sampled = metrics(frame, weights)
        except ValueError:
            continue
        for name in names:
            values[name].append(sampled[name])
    output: dict[str, float] = {}
    for name in names:
        low, high = np.quantile(values[name], (0.025, 0.975))
        output[f"{name}_ci_low"] = float(low)
        output[f"{name}_ci_high"] = float(high)
    return output


def class_metrics(frame: pd.DataFrame) -> list[dict[str, Any]]:
    target = frame.target.to_numpy(int)
    probability = probability_matrix(frame)
    prediction = probability.argmax(axis=1)
    rows = []
    for index, label in enumerate(CLASSES):
        positive = target == index
        called = prediction == index
        tp = int(np.logical_and(positive, called).sum())
        tn = int(np.logical_and(~positive, ~called).sum())
        fp = int(np.logical_and(~positive, called).sum())
        fn = int(np.logical_and(positive, ~called).sum())
        rows.append({
            "class": label,
            "n": int(positive.sum()),
            "auroc": float(roc_auc_score(positive.astype(int), probability[:, index])),
            "sensitivity": tp / max(tp + fn, 1),
            "specificity": tn / max(tn + fp, 1),
            "precision": float(precision_score(positive, called, zero_division=0)),
            "f1": float(f1_score(positive, called, zero_division=0)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        })
    return rows

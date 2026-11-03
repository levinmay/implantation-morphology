"""Evaluate training-only feature selection using fold-specific candidate tables."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, confusion_matrix
from pipeline.io import sha256_path, write_checksums
from paper_reproduction.experiments import stage_classifier as stage
from paper_reproduction.experiments.stage_classifier import CLASSES, probability_matrix
from paper_reproduction.experiments.feature_selection import fit_outer, validate_frame, SEED
from paper_reproduction.experiments.validation import validate_folds, validate_outer_frames

METHODS = {
    "whole_image_compact_appearance": ("whole", 9),
    "rich_mask_compact_morphology": ("rich", 6),
    "within_mask_compact_appearance": ("texture", 9),
    "compact_morphology_plus_appearance": ("combined", 15),
}
IDENTITY = ["unit_id", "component_id", "evaluation_fold", "target", "four_group"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def status(out, state, **extra):
    print(json.dumps({"status": state, **extra}), flush=True)


def load_features(directory):
    frames, whole_frames, hashes = {}, {}, {}
    for fold in range(5):
        folder = directory / f"outer_fold_{fold}" / "private"
        paths = {n: folder / n for n in ("PROVENANCE.json", "APPEARANCE_FEATURES.csv", "MORPHOLOGY_FEATURES.csv")}
        hashes.update({str(p): sha256_path(p) for p in paths.values()})
        provenance = json.loads(paths["PROVENANCE.json"].read_text())
        if (provenance.get("producer_outer_fold") != fold or
                provenance.get("outer_test_excluded_from_fitting_and_planning") is not True):
            raise ValueError("Missing upstream fold-isolation verification")
        appearance = pd.read_csv(paths["APPEARANCE_FEATURES.csv"])
        morphology = pd.read_csv(paths["MORPHOLOGY_FEATURES.csv"])
        rich = sorted(c for c in morphology if c.startswith("rich_"))
        texture = sorted(c for c in appearance if c.startswith("texture_"))
        if len(rich) != 49 or len(texture) != 42:
            raise ValueError("Extract all 91 candidates with --full-feature-pool")
        if set(morphology.producer_outer_fold) != {fold}:
            raise ValueError("Morphology has the wrong mask producer")
        for source in ("whole_crop", "nnunet_250_topology"):
            frame = appearance.loc[appearance.mask_source == source].copy()
            frame["four_group"] = frame.apply(lambda r: stage.four_group(r.parent_label, r.pre_substage), axis=1)
            frame["target"] = frame.four_group.map(CLASSES.index)
            validate_folds(frame, 577, 123)
            if set(frame.producer_outer_fold) != {fold}:
                raise ValueError("Appearance has the wrong mask producer")
            if frame.four_group.value_counts().to_dict() != stage.EXPECTED_COUNTS:
                raise ValueError("Unexpected class counts")
            if source == "nnunet_250_topology":
                if set(morphology.unit_id) != set(frame.unit_id):
                    raise ValueError("Morphology and appearance identities differ")
                frame = frame.merge(morphology[["unit_id", *rich]], on="unit_id", validate="one_to_one")
                frames[fold] = frame.sort_values("unit_id").reset_index(drop=True)
            else:
                whole_frames[fold] = frame.sort_values("unit_id").reset_index(drop=True)
    validate_outer_frames(frames)
    validate_outer_frames(whole_frames)
    whole = whole_frames[0]
    for fold in range(5):
        if not frames[fold][IDENTITY].equals(whole[IDENTITY]):
            raise ValueError("Whole-image and masked rows differ")
        if not np.allclose(whole[texture], whole_frames[fold][texture], equal_nan=True):
            raise ValueError("Whole-image features changed between producers")
    return frames, whole, sorted(rich + texture), hashes


def confusion_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    y = frame.target.to_numpy(int)
    prediction = probability_matrix(frame).argmax(axis=1)
    matrix = confusion_matrix(y, prediction, labels=np.arange(len(CLASSES)))
    rows = []
    for actual_index, actual in enumerate(CLASSES):
        for predicted_index, predicted in enumerate(CLASSES):
            rows.append({
                "actual": actual,
                "predicted": predicted,
                "count": int(matrix[actual_index, predicted_index]),
                "row_fraction": float(matrix[actual_index, predicted_index] / max(matrix[actual_index].sum(), 1)),
            })
    return rows

def small_metrics(frame, classes, weights=None):
    y = frame.target.to_numpy(int)
    p = frame[['prob_' + c for c in classes]].to_numpy(float)
    hard = p.argmax(axis=1)
    return {'macro_auroc': float(roc_auc_score(y, p, multi_class='ovr', average='macro',
                        labels=np.arange(4), sample_weight=weights)),
            'balanced_accuracy': float(balanced_accuracy_score(y, hard, sample_weight=weights)),
            'macro_f1': float(f1_score(y, hard, average='macro', sample_weight=weights))}

def summarize(out, stage, old_predictions):
    new = {}
    selected = []
    for method in METHODS:
        frame = pd.concat([pd.read_csv(out / f'private/{method}/fold_{k}/predictions.csv')
                           for k in range(5)], ignore_index=True).sort_values('unit_id').reset_index(drop=True)
        validate_frame(frame)
        if not frame[IDENTITY].equals(old_predictions[method][IDENTITY]):
            raise ValueError('Incomplete or unpaired final predictions')
        if not np.allclose(frame.filter(like='prob_').sum(axis=1), 1):
            raise ValueError('Invalid probabilities')
        frame.to_csv(out / f'private/predictions_{method}.csv', index=False)
        new[method] = frame
        for k in range(5):
            chosen = json.loads((out / f'private/{method}/fold_{k}/selection.json').read_text())
            selected.append({'method': method, **chosen})
    pd.DataFrame([{**r, 'features': json.dumps(r['features']),
                   'excluded_constant_or_missing': json.dumps(r['excluded_constant_or_missing'])}
                  for r in selected]).to_csv(out / 'aggregate/SELECTED_FEATURES_AND_MODELS.csv', index=False)
    for filename, fn in [('PER_CLASS.csv', stage.class_metrics), ('CONFUSION_MATRIX.csv', confusion_rows)]:
        pd.DataFrame([{'method': name, **r} for name, f in new.items() for r in fn(f)]).to_csv(
            out / 'aggregate' / filename, index=False)
    focus = 'compact_morphology_plus_appearance'
    all_frames = {('new', n): f for n, f in new.items()}
    all_frames.update({('old', n): f for n, f in old_predictions.items()})
    draws = {key: {m: [] for m in ['macro_auroc', 'balanced_accuracy', 'macro_f1']} for key in all_frames}
    for b, weights in enumerate(stage.grouped_weights(new[focus], 5000, SEED), 1):
        # Skip a draw for every method together if a bootstrap sample omits a class.
        if len(np.unique(new[focus].target.to_numpy()[weights > 0])) < 4:
            continue
        for key, frame in all_frames.items():
            for metric, value in small_metrics(frame, stage.CLASSES, weights).items():
                draws[key][metric].append(value)
        if b % 500 == 0:
            status(out, 'bootstrapping', completed=b, total=5000)
    summaries = []
    for name, frame in new.items():
        row = {'method': name, **stage.metrics(frame)}
        for metric, values in draws['new', name].items():
            if len(values) < 4500:
                raise ValueError('Too few usable bootstrap replicates')
            lo, hi = np.quantile(values, [.025, .975])
            row.update({metric + '_ci_low': float(lo), metric + '_ci_high': float(hi)})
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(out / 'aggregate/RESULTS.csv', index=False)
    contrasts = []
    pairs = [('reselected_vs_original', ('new', n), ('old', n)) for n in METHODS]
    pairs += [('combined_vs_representation', ('new', focus), ('new', n)) for n in METHODS if n != focus]
    for label, a, b in pairs:
        am, bm = small_metrics(all_frames[a], stage.CLASSES), small_metrics(all_frames[b], stage.CLASSES)
        for metric in am:
            delta = np.asarray(draws[a][metric]) - np.asarray(draws[b][metric])
            lo, hi = np.quantile(delta, [.025, .975])
            contrasts.append({'comparison': label, 'method': b[1], 'metric': metric,
                'difference': am[metric] - bm[metric], 'ci_low': float(lo), 'ci_high': float(hi)})
    pd.DataFrame(contrasts).to_csv(out / 'aggregate/PAIRED_COMPARISONS.csv', index=False)
    return summaries

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-feature-dir", type=Path, required=True)
    parser.add_argument("--fixed-predictions-dir", type=Path, required=True,
                        help="Private predictions from the fixed-feature comparison")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frames, whole, pool, hashes = load_features(args.outer_feature_dir)
    reference = whole[IDENTITY]
    fixed = {}
    for method in METHODS:
        path = args.fixed_predictions_dir / f"predictions_{method}.csv"
        hashes[str(path)] = sha256_path(path)
        frame = pd.read_csv(path).sort_values("unit_id").reset_index(drop=True)
        if not frame[IDENTITY].equals(reference):
            raise ValueError("Fixed-feature predictions do not match the selection cohort")
        fixed[method] = frame
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    (out / "aggregate").mkdir()
    (out / "private").mkdir()
    for fold in range(5):
        for method, (family, budget) in METHODS.items():
            columns = pool if family == "combined" else [c for c in pool if c.startswith(
                "rich_" if family == "rich" else "texture_")]
            frame = whole if family == "whole" else frames[fold]
            status(out, "ranking_and_fitting", outer_fold=fold, method=method)
            pred, record, ranking, detail, trials, estimator = fit_outer(
                frame, columns, budget, fold, stage.candidates, CLASSES)
            dest = out / "private" / method / f"fold_{fold}"
            dest.mkdir(parents=True)
            pred.to_csv(dest / "predictions.csv", index=False)
            ranking.to_csv(dest / "ranking.csv", index=False)
            detail.to_csv(dest / "inner_importance.csv", index=False)
            pd.DataFrame(trials).to_csv(dest / "validation_trials.csv", index=False)
            write_json(dest / "selection.json", record)
            joblib.dump({"estimator": estimator, "features": record["features"],
                         "classes": CLASSES, "outer_fold": fold}, dest / "model.joblib")
    summarize(out, stage, fixed)
    if any(sha256_path(Path(p)) != digest for p, digest in hashes.items()):
        raise ValueError("Inputs changed during evaluation")
    write_json(out / "private/INPUTS.json", hashes)
    write_json(out / "aggregate/VALIDATION.json", {
        "status": "PASS", "units": 577, "outer_folds": 5, "components": 123,
        "training_only_feature_selection": True, "new_segmentation_training": False})
    write_checksums(out)


if __name__ == "__main__":
    main()

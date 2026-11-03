"""Check fold assignments and feature lists."""

import pandas as pd
from pipeline.features import validate_feature_set


def validate_folds(frame: pd.DataFrame, expected_units: int | None = None,
                   expected_components: int | None = None) -> None:
    required = {"unit_id", "component_id", "evaluation_fold"}
    if not required.issubset(frame):
        raise ValueError(f"Missing fold columns: {sorted(required - set(frame))}")
    if frame[list(required)].isna().any().any():
        raise ValueError("Missing unit, component, or fold identifier")
    if frame.unit_id.duplicated().any():
        raise ValueError("Duplicate unit identifiers")
    if expected_units is not None and len(frame) != expected_units:
        raise ValueError(f"Expected {expected_units} units")
    if set(frame.evaluation_fold) != set(range(5)):
        raise ValueError("Expected all five evaluation folds, numbered 0--4")
    if (frame.groupby("component_id").evaluation_fold.nunique() != 1).any():
        raise ValueError("A linkage component crosses evaluation folds")
    if expected_components is not None and frame.component_id.nunique() != expected_components:
        raise ValueError(f"Expected {expected_components} linkage components")


def validate_outer_frames(frames: dict[int, pd.DataFrame]) -> None:
    """Reject mixed mask producers and changed identities across outer caches."""
    if set(frames) != set(range(5)):
        raise ValueError("Expected five outer-fold feature caches")
    identity = ["unit_id", "component_id", "evaluation_fold", "target", "four_group"]
    reference = None
    for fold, frame in sorted(frames.items()):
        validate_folds(frame)
        if "producer_outer_fold" not in frame or set(frame.producer_outer_fold) != {fold}:
            raise ValueError("Every row must use the matching outer-fold mask producer")
        current = frame[identity].sort_values("unit_id").reset_index(drop=True)
        if reference is not None and not current.equals(reference):
            raise ValueError("Outer-fold feature identities differ")
        reference = current


def validate_upstream_split(rows: list[dict], split: dict, fold: int) -> None:
    """Check recorded fitting/planning roles against actual nnU-Net splits."""
    by_case = {r["case_id"]: r for r in rows}
    if len(by_case) != len(rows):
        raise ValueError("Duplicate upstream case IDs")
    train, validation = set(split["train"]), set(split["val"])
    test = {r["case_id"] for r in rows if int(r["source_evaluation_fold"]) == fold}
    if (train & validation or (train | validation) & test
            or train | validation | test != set(by_case)):
        raise ValueError("Upstream split is not a disjoint complete partition")
    for cases, role in ((train, "train"), (validation, "val"), (test, "test")):
        if cases != {r["case_id"] for r in rows if r["role"] == role}:
            raise ValueError("Upstream split differs from recorded roles")
    for field in ("component_id", "embryo_id"):
        if {by_case[c][field] for c in test} & {by_case[c][field] for c in train | validation}:
            raise ValueError(f"Outer-test overlap in {field}")
    # Older case tables store source linkage in component_id, not a source hash.
    if all(r.get("source_sha256") for r in rows):
        if {by_case[c]["source_sha256"] for c in test} & {
                by_case[c]["source_sha256"] for c in train | validation}:
            raise ValueError("Outer-test overlap in source_sha256")

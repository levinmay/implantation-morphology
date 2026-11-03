"""Refuse unverified or changed nnU-Net preparation and prediction caches."""

import json
from pathlib import Path

from pipeline.io import sha256_path
from pipeline.provenance import software_identity


def source_identity(args, fold: int, manifest: list[dict]) -> dict:
    return {
        "fold": fold, "dataset_id": args.dataset_id_base + fold,
        "manifest_sha256": sha256_path(args.release_dir / "tables/embryo_mask_manifest.csv"),
        "assignments_sha256": sha256_path(args.split_dir / "unit_fold_assignment.csv"),
        "inputs": {row["unit_id"]: {
            "image": sha256_path(args.project_root / row["crop_path"]),
            "mask": sha256_path(args.release_dir / row["mask_path"]),
        } for row in manifest},
    }


def prepared_artifacts(paths: dict) -> dict:
    files = [paths["root"] / "case_table.csv"]
    for key in ("raw_dataset", "test_input", "test_labels"):
        files.extend(p for p in paths[key].rglob("*") if p.is_file())
    # nnU-Net may later create unpacked training caches; they are not preparation metadata.
    files.extend(paths["preprocessed_dataset"].glob("*.json"))
    return {p.relative_to(paths["root"]).as_posix(): sha256_path(p) for p in sorted(files)}


def preparation_record(paths: dict, sources: dict) -> dict:
    runner = Path(__file__).with_name("run_embryo577_nnunet_benchmark.py")
    return {"schema_version": 1, "sources": sources, "artifacts": prepared_artifacts(paths),
            "runner_sha256": sha256_path(runner), "guard_sha256": sha256_path(Path(__file__)),
            "software": software_identity()}


def verify_preparation(paths: dict, sources: dict) -> None:
    path = paths["root"] / "PREPARATION_PROVENANCE.json"
    if not path.is_file() or json.loads(path.read_text()) != preparation_record(paths, sources):
        raise ValueError("Preparation inputs, metadata, or software changed/unverified; use a new output directory")


def prediction_context(paths: dict, checkpoint: Path, seed: int) -> dict:
    return {"checkpoint_sha256": sha256_path(checkpoint), "seed": seed,
            "prepare_provenance_sha256": sha256_path(paths["root"] / "PREPARATION_PROVENANCE.json"),
            "case_table_sha256": sha256_path(paths["root"] / "case_table.csv"),
            "test_images": {p.name: sha256_path(p) for p in sorted(paths["test_input"].glob("*.png"))},
            "model_metadata": {p.name: sha256_path(p) for p in sorted(checkpoint.parent.parent.glob("*.json"))},
            "software": software_identity(), "tta": True, "configuration": "2d"}


def mask_hashes(paths: dict) -> dict:
    return {p.name: sha256_path(p) for p in sorted(paths["raw_predictions"].glob("*.png"))}


def verify_predictions(paths: dict, context: dict, expected: set[str]) -> bool:
    files = mask_hashes(paths)
    record = paths["root"] / "PREDICTION_PROVENANCE.json"
    if not files and not record.exists():
        return False
    if not record.is_file():
        raise ValueError("Existing predictions are unverified; use a new output directory")
    saved = json.loads(record.read_text())
    if (saved.get("context") != context or saved.get("mask_sha256") != files
            or set(files) != expected):
        raise ValueError("Prediction checkpoint, inputs, or masks changed; use a new output directory")
    return True

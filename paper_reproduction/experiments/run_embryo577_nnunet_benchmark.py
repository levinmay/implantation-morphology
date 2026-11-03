#!/usr/bin/env python3
"""Prepare, train, predict, and score the Embryo-577 nnU-Net benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any
from paper_reproduction.experiments.validation import validate_upstream_split
from paper_reproduction.experiments.nnunet_provenance import (
    source_identity, preparation_record, verify_preparation,
    prediction_context, mask_hashes, verify_predictions,
)

import numpy as np
import pandas as pd
from PIL import Image
from pipeline.io import sha256_path
from pipeline.cleanup import topology

from paper_reproduction.experiments.study import (
    EXPECTED_UNITS,
    load_authority,
    aggregate_rows,
    four_group,
    read_csv,
    score_mask,
    write_csv,
    write_json,
)


TRAINER = "nnUNetTrainer_250epochs_seeded"
CONFIGURATION = "2d"
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", type=Path, required=True)
    common.add_argument("--release-dir", type=Path, required=True)
    common.add_argument("--split-dir", type=Path, required=True)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--dataset-id-base", type=int, default=986)
    common.add_argument("--seed", type=int, default=SEED)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", parents=[common])
    prepare.add_argument("--fold", type=int, choices=range(5), action="append")
    prepare.add_argument("--num-processes", type=int, default=2)
    run = sub.add_parser("run-fold", parents=[common])
    run.add_argument("--fold", type=int, choices=range(5), required=True)
    run.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    aggregate = sub.add_parser("aggregate", parents=[common])
    staging = sub.add_parser("predict-staging", parents=[common])
    staging.add_argument("--fold", type=int, choices=range(5), required=True)
    staging.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    staging.add_argument("--staging-mask-dir", type=Path, required=True)
    return parser.parse_args()


def case_id(unit_id: str) -> str:
    return unit_id


def fold_paths(output_dir: Path, dataset_id: int, fold: int) -> dict[str, Path]:
    root = output_dir / "folds" / f"fold_{fold}"
    env = root / "nnunet_env"
    dataset_name = f"Dataset{dataset_id:03d}_embryo577_fold{fold}"
    return {
        "root": root,
        "env": env,
        "raw": env / "raw",
        "preprocessed": env / "preprocessed",
        "results": env / "results",
        "dataset_name": Path(dataset_name),
        "raw_dataset": env / "raw" / dataset_name,
        "preprocessed_dataset": env / "preprocessed" / dataset_name,
        "test_input": root / "test_input",
        "test_labels": root / "test_labels",
        "raw_predictions": root / "raw_predictions",
    }


def command_env(paths: dict[str, Path], seed: int = SEED) -> dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(paths["raw"])
    env["nnUNet_preprocessed"] = str(paths["preprocessed"])
    env["nnUNet_results"] = str(paths["results"])
    env["MPLCONFIGDIR"] = str(paths["env"] / ".mplconfig")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["NNUNET_SEED"] = str(seed)
    return env


def binary(name: str) -> Path:
    # sys.prefix preserves the active venv; resolving its Python symlink does not.
    value = Path(os.sys.prefix) / "bin" / name
    if not value.is_file():
        raise FileNotFoundError(value)
    return value


def load_authorities(
    release_dir: Path,
    split_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    return load_authority(release_dir, split_dir)


def roles_for_fold(
    manifest: list[dict[str, str]],
    assignment_map: dict[str, dict[str, str]],
    fold: int,
) -> dict[str, str]:
    result = {}
    for row in manifest:
        unit_id = row["unit_id"]
        evaluation_fold = int(assignment_map[unit_id]["evaluation_fold"])
        if evaluation_fold == fold:
            role = "test"
        elif evaluation_fold == (fold + 1) % 5:
            role = "val"
        else:
            role = "train"
        result[unit_id] = role
    return result


def save_rgb(source: Path, destination: Path) -> tuple[int, int]:
    image = Image.open(source).convert("RGB")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return image.height, image.width


def save_mask(source: Path, destination: Path, expected_shape: tuple[int, int]) -> None:
    mask = np.asarray(Image.open(source).convert("L")) > 0
    if mask.shape != expected_shape:
        raise RuntimeError(f"Mask/image native-grid mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8), mode="L").save(destination)


def prepare_fold(args: argparse.Namespace, fold: int) -> None:
    project_root = args.project_root.resolve()
    release_dir = args.release_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    dataset_id = args.dataset_id_base + fold
    paths = fold_paths(output_dir, dataset_id, fold)
    status_path = paths["root"] / "PREPARE_STATUS.json"
    manifest, assignment_map = load_authorities(release_dir, split_dir)
    sources = source_identity(args, fold, manifest)
    if status_path.is_file():
        status = json.loads(status_path.read_text())
        if status.get("status") == "PASS":
            verify_preparation(paths, sources)
            print(f"fold {fold} already prepared", flush=True)
            return
    if paths["root"].exists():
        raise FileExistsError(f"Incomplete fold workspace exists: {paths['root']}")

    from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
    roles = roles_for_fold(manifest, assignment_map, fold)
    paths["root"].mkdir(parents=True)
    for key in ("raw", "preprocessed", "results", "test_input", "test_labels"):
        paths[key].mkdir(parents=True, exist_ok=True)
    images_tr = paths["raw_dataset"] / "imagesTr"
    labels_tr = paths["raw_dataset"] / "labelsTr"
    images_tr.mkdir(parents=True)
    labels_tr.mkdir(parents=True)

    case_rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest, start=1):
        unit_id = row["unit_id"]
        identifier = case_id(unit_id)
        role = roles[unit_id]
        image_source = project_root / row["crop_path"]
        mask_source = release_dir / row["mask_path"]
        if role in {"train", "val"}:
            image_destination = images_tr / f"{identifier}_0000.png"
            label_destination = labels_tr / f"{identifier}.png"
        else:
            image_destination = paths["test_input"] / f"{identifier}_0000.png"
            label_destination = paths["test_labels"] / f"{identifier}.png"
        shape = save_rgb(image_source, image_destination)
        save_mask(mask_source, label_destination, shape)
        split = assignment_map[unit_id]
        case_rows.append(
            {
                "unit_id": unit_id,
                "case_id": identifier,
                "role": role,
                "evaluation_fold": fold,
                "source_evaluation_fold": split["evaluation_fold"],
                "component_id": split["component_id"],
                "embryo_id": row["embryo_id"],
                "experiment": row["experiment"],
                "dpf": row["dpf"],
                "four_group": four_group(row),
                "height": shape[0],
                "width": shape[1],
            }
        )
        if index % 100 == 0:
            print(f"fold {fold} export {index}/{EXPECTED_UNITS}", flush=True)

    counts = Counter(roles.values())
    generate_dataset_json(
        str(paths["raw_dataset"]),
        channel_names={0: "R", 1: "G", 2: "B"},
        labels={"background": 0, "embryo": 1},
        num_training_cases=counts["train"] + counts["val"],
        file_ending=".png",
        dataset_name=str(paths["dataset_name"]),
        description="Embryo-577 leakage-safe whole-embryo RGB segmentation benchmark.",
        overwrite_image_reader_writer="NaturalImage2DIO",
        converted_by="Anonymous authors",
    )
    splits = [
        {
            "train": [case_id(unit_id) for unit_id, role in roles.items() if role == "train"],
            "val": [case_id(unit_id) for unit_id, role in roles.items() if role == "val"],
        }
    ]
    write_json(paths["raw_dataset"] / "splits_final.json", splits)
    write_csv(paths["root"] / "case_table.csv", case_rows)
    write_json(
        paths["root"] / "PREPARE_STATUS.json",
        {"status": "planning", "fold": fold, "dataset_id": dataset_id, "role_counts": dict(counts)},
    )
    command = [
        str(binary("nnUNetv2_plan_and_preprocess")),
        "-d", str(dataset_id), "-c", CONFIGURATION,
        "-np", str(args.num_processes), "--verify_dataset_integrity",
    ]
    subprocess.check_call(command, env=command_env(paths))
    if not paths["preprocessed_dataset"].is_dir():
        raise RuntimeError(f"Missing preprocessed dataset: {paths['preprocessed_dataset']}")
    write_json(paths["preprocessed_dataset"] / "splits_final.json", splits)
    if source_identity(args, fold, manifest) != sources:
        raise ValueError("Source inputs changed during preparation")
    write_json(paths["root"] / "PREPARATION_PROVENANCE.json", preparation_record(paths, sources))
    write_json(
        status_path,
        {
            "status": "PASS",
            "fold": fold,
            "dataset_id": dataset_id,
            "dataset_name": str(paths["dataset_name"]),
            "role_counts": dict(counts),
            "rgb_native_grid": True,
            "split_assignment_sha256": sha256_path(split_dir / "unit_fold_assignment.csv"),
            "release_manifest_sha256": sha256_path(
                release_dir / "tables/embryo_mask_manifest.csv"
            ),
            "splits_final_sha256": sha256_path(paths["preprocessed_dataset"] / "splits_final.json"),
        },
    )
    print(json.dumps({"fold": fold, "status": "PASS", "roles": dict(counts)}), flush=True)


def trainer_dir(paths: dict[str, Path]) -> Path:
    return (
        paths["results"] / str(paths["dataset_name"])
        / f"{TRAINER}__nnUNetPlans__{CONFIGURATION}" / "fold_0"
    )


def run_fold(args: argparse.Namespace) -> None:
    fold = args.fold
    output_dir = args.output_dir.resolve()
    dataset_id = args.dataset_id_base + fold
    paths = fold_paths(output_dir, dataset_id, fold)
    status_path = paths["root"] / "PREPARE_STATUS.json"
    if not status_path.is_file() or json.loads(status_path.read_text()).get("status") != "PASS":
        raise RuntimeError(f"Fold {fold} preparation has not passed")
    manifest, _ = load_authorities(args.release_dir, args.split_dir)
    verify_preparation(paths, source_identity(args, fold, manifest))
    run_status_path = paths["root"] / "RUN_STATUS.json"
    if run_status_path.exists():
        recorded_seed = json.loads(run_status_path.read_text()).get("seed", SEED)
        if recorded_seed != args.seed:
            raise ValueError("Use a separate output directory for each training seed")
    model_dir = trainer_dir(paths)
    checkpoint_final = model_dir / "checkpoint_final.pth"
    start = time.time()
    if not checkpoint_final.is_file():
        command = [
            str(binary("nnUNetv2_train")), str(dataset_id), CONFIGURATION, "0",
            "-tr", TRAINER, "-device", args.device,
        ]
        if (model_dir / "checkpoint_latest.pth").is_file():
            command.append("--c")
        write_json(
            run_status_path,
            {"status": "training", "fold": fold, "seed": args.seed,
             "command": command, "started": start},
        )
        subprocess.check_call(command, env=command_env(paths, args.seed))
    if not checkpoint_final.is_file():
        raise RuntimeError(f"Fold {fold} did not produce checkpoint_final.pth")

    case_rows = read_csv(paths["root"] / "case_table.csv")
    test_rows = [row for row in case_rows if row["role"] == "test"]
    paths["raw_predictions"].mkdir(parents=True, exist_ok=True)
    context = prediction_context(paths, checkpoint_final, args.seed)
    expected_files = {f"{row['case_id']}.png" for row in test_rows}
    if not verify_predictions(paths, context, expected_files):
        command = [
            str(binary("nnUNetv2_predict")),
            "-i", str(paths["test_input"]),
            "-o", str(paths["raw_predictions"]),
            "-d", str(dataset_id), "-tr", TRAINER, "-c", CONFIGURATION,
            "-f", "0", "-device", args.device, "-chk", "checkpoint_final.pth",
            "-npp", "1", "-nps", "1",
        ]
        subprocess.check_call(command, env=command_env(paths))
    predictions = {path.stem for path in paths["raw_predictions"].glob("*.png")}
    expected = {row["case_id"] for row in test_rows}
    if predictions != expected:
        raise RuntimeError(
            f"Fold {fold} prediction IDs differ: {len(predictions)} vs {len(expected)}"
        )
    if prediction_context(paths, checkpoint_final, args.seed) != context:
        raise ValueError("Checkpoint or prediction inputs changed during inference")
    write_json(paths["root"] / "PREDICTION_PROVENANCE.json", {
        "schema_version": 1, "context": context, "mask_sha256": mask_hashes(paths),
    })

    metric_rows: list[dict[str, Any]] = []
    for row in test_rows:
        identifier = row["case_id"]
        gt = np.asarray(Image.open(paths["test_labels"] / f"{identifier}.png").convert("L")) > 0
        prediction = np.asarray(
            Image.open(paths["raw_predictions"] / f"{identifier}.png").convert("L")
        ) > 0
        if prediction.shape != gt.shape:
            raise RuntimeError(f"Native-grid prediction mismatch for {identifier}")
        item = {
            "gt": gt,
            "unit_id": row["unit_id"],
            "embryo_id": row["embryo_id"],
            "experiment": row["experiment"],
            "dpf": int(float(row["dpf"])),
            "four_group": row["four_group"],
            "component_id": row["component_id"],
            "evaluation_fold": fold,
            "anns": [],
        }
        metric_rows.append(score_mask(item, "nnunet_250", prediction, float("nan"), -1))
        cleaned = topology(prediction)
        metric_rows.append(score_mask(item, "nnunet_250_topology", cleaned, float("nan"), -1))
        cleaned_path = paths["root"] / "topology_predictions" / f"{identifier}.png"
        cleaned_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(cleaned.astype(np.uint8) * 255, mode="L").save(cleaned_path)
    write_csv(paths["root"] / "per_unit_metrics.csv", metric_rows)
    write_json(
        run_status_path,
        {
            "status": "PASS",
            "fold": fold,
            "test_units": len(test_rows),
            "tta_enabled": True,
            "seed": args.seed,
            "trainer": TRAINER,
            "checkpoint_final_sha256": context["checkpoint_sha256"],
            "elapsed_seconds": time.time() - start,
        },
    )
    print(json.dumps(json.loads(run_status_path.read_text()), indent=2), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        path = output_dir / "folds" / f"fold_{fold}" / "per_unit_metrics.csv"
        rows.extend(read_csv(path))
    frame = pd.DataFrame(rows)
    numeric = [
        "dpf", "evaluation_fold", "selection_score", "proposal_index", "proposal_count",
        "dice", "nsd_1pct", "normalized_hd95", "relative_area_error",
        "prediction_area_px", "reference_area_px",
    ]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    duplicates = frame.duplicated(["method", "unit_id"]).sum()
    counts = frame.groupby("method")["unit_id"].nunique().to_dict()
    if duplicates or any(value != EXPECTED_UNITS for value in counts.values()):
        raise RuntimeError(f"Invalid OOF coverage: duplicates={duplicates}, counts={counts}")
    typed_rows = frame.to_dict(orient="records")
    write_csv(output_dir / "results/nnunet_per_unit_metrics.csv", typed_rows)
    write_csv(output_dir / "results/nnunet_paper_table.csv", aggregate_rows(typed_rows))
    write_json(
        output_dir / "results/NNUNET_VALIDATION.json",
        {
            "status": "PASS",
            "units": EXPECTED_UNITS,
            "methods": counts,
            "duplicate_method_unit_rows": int(duplicates),
            "each_unit_tested_once": True,
            "native_grid": True,
            "tta_enabled": True,
        },
    )
    print(json.dumps(aggregate_rows(typed_rows), indent=2), flush=True)


def verify_staging_inputs(paths: dict[str, Path], fold: int) -> tuple[list[dict], dict]:
    rows = read_csv(paths["root"] / "case_table.csv")
    raw_split = json.loads((paths["raw_dataset"] / "splits_final.json").read_text())
    processed_split = json.loads((paths["preprocessed_dataset"] / "splits_final.json").read_text())
    if raw_split != processed_split or len(raw_split) != 1:
        raise ValueError("Raw and preprocessed upstream splits disagree")
    validate_upstream_split(rows, raw_split[0], fold)
    fitting = set(raw_split[0]["train"]) | set(raw_split[0]["val"])
    planned = {p.name.removesuffix("_0000.png")
               for p in (paths["raw_dataset"] / "imagesTr").glob("*_0000.png")}
    if planned != fitting:
        raise ValueError("Planning images differ from the upstream fitting partition")
    status = json.loads((paths["root"] / "RUN_STATUS.json").read_text())
    checkpoint = trainer_dir(paths) / "checkpoint_final.pth"
    if (status.get("status") != "PASS" or status.get("seed") != SEED
            or sha256_path(checkpoint) != status.get("checkpoint_final_sha256")):
        raise ValueError("Expected verified primary seed-42 checkpoint")
    expected = {f"{r['case_id']}.png" for r in rows if r["role"] == "test"}
    if not verify_predictions(paths, prediction_context(paths, checkpoint, SEED), expected):
        raise ValueError("Missing verified held-out predictions")
    provenance = {
        "producer_outer_fold": fold,
        "outer_test_excluded_from_fitting_and_planning": True,
        "checkpoint_final_sha256": sha256_path(checkpoint),
        "case_table_sha256": sha256_path(paths["root"] / "case_table.csv"),
        "split_sha256": sha256_path(paths["preprocessed_dataset"] / "splits_final.json"),
        "upstream_training_masks": "in-sample",
    }
    return rows, provenance


def predict_staging(args: argparse.Namespace) -> None:
    """Inference only: preserve held-out masks and infer all other rows with model k."""
    if args.seed != SEED:
        raise ValueError("The reported staging protocol uses primary seed 42 only")
    fold = args.fold
    paths = fold_paths(args.output_dir.resolve(), args.dataset_id_base + fold, fold)
    rows, provenance = verify_staging_inputs(paths, fold)
    target = args.staging_mask_dir.resolve() / f"outer_fold_{fold}"
    target.mkdir(parents=True, exist_ok=False)
    inputs, predictions = target / "input", target / "raw_predictions"
    inputs.mkdir()
    predictions.mkdir()
    for row in rows:
        name = row["case_id"]
        if row["role"] == "test":
            source = paths["raw_predictions"] / f"{name}.png"
            if not source.is_file():
                raise FileNotFoundError(source)
            (predictions / source.name).symlink_to(source)
        else:
            source = paths["raw_dataset"] / "imagesTr" / f"{name}_0000.png"
            (inputs / source.name).symlink_to(source)
    command = [str(binary("nnUNetv2_predict")), "-i", str(inputs), "-o", str(predictions),
               "-d", str(args.dataset_id_base + fold), "-tr", TRAINER, "-c", CONFIGURATION,
               "-f", "0", "-device", args.device, "-chk", "checkpoint_final.pth",
               "-npp", "1", "-nps", "1"]
    subprocess.check_call(command, env=command_env(paths))
    if {p.stem for p in predictions.glob("*.png")} != {r["case_id"] for r in rows}:
        raise ValueError("Incomplete staging mask set")
    provenance["mask_sha256"] = {p.stem: sha256_path(p) for p in sorted(predictions.glob("*.png"))}
    write_json(target / "PROVENANCE.json", provenance)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        folds = args.fold if args.fold is not None else list(range(5))
        for fold in folds:
            prepare_fold(args, fold)
    elif args.command == "run-fold":
        run_fold(args)
    elif args.command == "aggregate":
        aggregate(args)
    elif args.command == "predict-staging":
        predict_staging(args)


if __name__ == "__main__":
    main()

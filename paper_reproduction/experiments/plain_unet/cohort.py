"""Load the paper cohort and verify its segmentation splits."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from paper_reproduction.experiments.study import load_authority, four_group
from paper_reproduction.experiments.validation import validate_upstream_split

def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def save_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def status(output, state, **extra):
    record = {"status": state, "pid": os.getpid(), "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
    save_json(output / "STATUS.json", record)
    print(json.dumps(record), flush=True)


def partition(frame, outer_fold):
    validation_fold = (outer_fold + 1) % 5
    return {"train": frame.loc[~frame.evaluation_fold.isin([outer_fold, validation_fold])],
            "val": frame.loc[frame.evaluation_fold == validation_fold],
            "test": frame.loc[frame.evaluation_fold == outer_fold]}


def dice(prediction, reference):
    denominator = int(prediction.sum()) + int(reference.sum())
    return 2 * int(np.logical_and(prediction, reference).sum()) / denominator if denominator else 0.0


def prepare(args):
    manifest, assignments = load_authority(args.release_dir, args.split_dir)
    sources = {}
    def witness(path):
        path = Path(path).resolve()
        sources[str(path)] = digest(path)
        return path
    witness(args.release_dir / "tables/embryo_mask_manifest.csv")
    witness(args.split_dir / "unit_fold_assignment.csv")
    case_maps = {}
    for fold in range(5):
        folder = args.nnunet_dir / "folds" / f"fold_{fold}"
        cases = pd.read_csv(witness(folder / "case_table.csv")).to_dict("records")
        split_files = list((folder / "nnunet_env/preprocessed").glob("*/splits_final.json"))
        if len(split_files) != 1:
            raise ValueError("Expected one recorded nnU-Net split per fold")
        split = json.loads(witness(split_files[0]).read_text())[0]
        validate_upstream_split(cases, split, fold)
        by_unit = {r["unit_id"]: r for r in cases}
        if set(by_unit) != set(assignments):
            raise ValueError("nnU-Net cases and manifest differ")
        for unit, case in by_unit.items():
            assignment = assignments[unit]
            expected_role = ("test" if int(assignment["evaluation_fold"]) == fold else
                             "val" if int(assignment["evaluation_fold"]) == (fold + 1) % 5 else "train")
            if (case["role"] != expected_role or
                    int(case["source_evaluation_fold"]) != int(assignment["evaluation_fold"]) or
                    str(case["component_id"]) != str(assignment["component_id"])):
                raise ValueError("U-Net and nnU-Net folds differ")
        case_maps[fold] = by_unit
    rows = []
    for row in manifest:
        unit = row["unit_id"]
        assignment = assignments[unit]
        fold = int(assignment["evaluation_fold"])
        case = case_maps[fold][unit]
        image = witness(args.project_root / row["crop_path"])
        mask = witness(args.release_dir / row["mask_path"])
        prediction = witness(args.nnunet_dir / f"folds/fold_{fold}/raw_predictions/{case['case_id']}.png")
        for path, field in ((image, "crop_sha256"), (mask, "mask_sha256")):
            if row.get(field) and sources[str(path)] != row[field]:
                raise ValueError("Image or mask differs from its manifest hash")
        rows.append({"unit_id": unit, "component_id": assignment["component_id"],
                     "evaluation_fold": fold, "embryo_id": row["embryo_id"],
                     "image": str(image), "mask": str(mask), "nnunet_prediction": str(prediction),
                     "four_group": four_group(row)})
    frame = pd.DataFrame(rows).sort_values("unit_id").reset_index(drop=True)
    if frame.embryo_id.nunique() != 131:
        raise ValueError("Expected 131 embryos")
    if (frame.groupby("embryo_id").evaluation_fold.nunique() != 1).any():
        raise ValueError("An embryo crosses folds")
    save_json(args.output_dir / "private/INPUTS.json", {"sources": sources, "units": len(frame)})
    return frame, sources

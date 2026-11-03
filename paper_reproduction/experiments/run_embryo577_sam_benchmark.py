#!/usr/bin/env python3
"""Generate and evaluate native-grid SAM proposals for Embryo-577."""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pipeline.io import sha256_path
from paper_reproduction.experiments.study import (
    EXPECTED_UNITS, GROUPS, load_authority, four_group, score_mask,
    aggregate_rows, write_csv, write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-root", type=Path, required=True)
    common.add_argument("--release-dir", type=Path, required=True)
    common.add_argument("--split-dir", type=Path, required=True)
    common.add_argument("--output-dir", type=Path, required=True)
    generate = sub.add_parser("generate", parents=[common])
    generate.add_argument("--sam-checkpoint", type=Path, required=True)
    generate.add_argument("--mode", choices=("smoke", "full"), required=True)
    generate.add_argument("--points-per-side", type=int, default=32)
    generate.add_argument("--pred-iou-thresh", type=float, default=0.88)
    generate.add_argument("--stability-thresh", type=float, default=0.95)
    generate.add_argument("--points-per-batch", type=int, default=64)
    evaluate = sub.add_parser("evaluate", parents=[common])
    evaluate.add_argument("--mode", choices=("smoke", "full"), required=True)
    return parser.parse_args()


def smoke_units(
    manifest: list[dict[str, str]],
    assignment_map: dict[str, dict[str, str]],
) -> list[str]:
    chosen: list[str] = []
    for group in GROUPS:
        row = next((value for value in manifest if four_group(value) == group), None)
        if row is not None:
            chosen.append(row["unit_id"])
    for fold in range(5):
        row = next(
            value for value in manifest
            if int(assignment_map[value["unit_id"]]["evaluation_fold"]) == fold
        )
        chosen.append(row["unit_id"])
    return sorted(set(chosen))


def cache_path(output_dir: Path, unit_id: str) -> Path:
    return output_dir / "cache" / "items" / f"{unit_id}.pkl.gz"


def load_cache(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def save_cache(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=5) as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def validate_cache_configuration(output_dir: Path, configuration: dict) -> None:
    saved = output_dir / "SAM_CONFIGURATION.json"
    if saved.is_file():
        if json.loads(saved.read_text()) != configuration:
            raise ValueError("SAM configuration changed; use a new output directory")
    elif any((output_dir / "cache/items").glob("*.pkl.gz")):
        raise ValueError("Cached proposals have no configuration; use a new output directory")


def generate(args: argparse.Namespace) -> None:
    import torch
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SAM generation")
    project_root = args.project_root.resolve()
    release_dir = args.release_dir.resolve()
    split_dir = args.split_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest, assignment_map = load_authority(release_dir, split_dir)
    selected = (
        smoke_units(manifest, assignment_map)
        if args.mode == "smoke"
        else [row["unit_id"] for row in manifest]
    )
    manifest_map = {row["unit_id"]: row for row in manifest}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.sam_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    configuration = {
        "mode": args.mode,
        "model_type": "vit_h",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_path(checkpoint),
        "points_per_side": args.points_per_side,
        "points_per_batch": args.points_per_batch,
        "pred_iou_thresh": args.pred_iou_thresh,
        "stability_score_thresh": args.stability_thresh,
        "box_nms_thresh": 0.7,
        "crop_n_layers": 0,
        "crop_nms_thresh": 0.7,
        "crop_overlap_ratio": 512 / 1500,
        "crop_n_points_downscale_factor": 1,
        "min_mask_region_area": 0,
        "output_mode": "binary_mask",
        "input_grid": "native identity-safe crop; no manual resize or padding",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "release_manifest_sha256": sha256_path(
            release_dir / "tables/embryo_mask_manifest.csv"
        ),
        "split_assignment_sha256": sha256_path(split_dir / "unit_fold_assignment.csv"),
    }
    validate_cache_configuration(output_dir, configuration)
    write_json(output_dir / "SAM_CONFIGURATION.json", configuration)
    write_json(output_dir / "RUN_STATUS.json", {"status": "running", "started": time.time()})

    sam = sam_model_registry["vit_h"](checkpoint=str(checkpoint)).to(torch.device("cuda"))
    generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_thresh,
        box_nms_thresh=0.7,
        crop_n_layers=0,
        crop_nms_thresh=0.7,
        crop_overlap_ratio=512 / 1500,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=0,
        output_mode="binary_mask",
    )
    cache_rows: list[dict[str, Any]] = []
    start = time.time()
    for index, unit_id in enumerate(selected, start=1):
        path = cache_path(output_dir, unit_id)
        row = manifest_map[unit_id]
        crop_path = project_root / row["crop_path"]
        gt_path = release_dir / row["mask_path"]
        input_hashes = {"image": sha256_path(crop_path), "mask": sha256_path(gt_path)}
        if path.is_file():
            item = load_cache(path)
            if item.get("input_sha256") != input_hashes:
                raise ValueError("Cached proposal inputs changed or are unverified; use a new directory")
        else:
            rgb = np.asarray(Image.open(crop_path).convert("RGB"), dtype=np.uint8)
            gt = np.asarray(Image.open(gt_path).convert("L"), dtype=np.uint8) > 0
            if gt.shape != rgb.shape[:2]:
                raise RuntimeError(f"Native-grid mismatch for {unit_id}")
            annotations = generator.generate(rgb)
            for annotation in annotations:
                annotation["segmentation"] = annotation["segmentation"].astype(bool)
            split = assignment_map[unit_id]
            item = {
                "input_sha256": input_hashes,
                "gt": gt,
                "H": int(gt.shape[0]),
                "W": int(gt.shape[1]),
                "anns": annotations,
                "image_id": unit_id,
                "unit_id": unit_id,
                "embryo_id": row["embryo_id"],
                "experiment": row["experiment"],
                "dpf": int(float(row["dpf"])),
                "four_group": four_group(row),
                "component_id": split["component_id"],
                "evaluation_fold": int(split["evaluation_fold"]),
            }
            save_cache(path, item)
        cache_rows.append(
            {
                "unit_id": unit_id,
                "cache_path": str(path.relative_to(output_dir)),
                "cache_sha256": sha256_path(path),
                "height": item["H"],
                "width": item["W"],
                "proposal_count": len(item["anns"]),
                "evaluation_fold": item["evaluation_fold"],
                "four_group": item["four_group"],
            }
        )
        if index % 10 == 0 or index == len(selected):
            print(
                json.dumps(
                    {
                        "processed": index,
                        "total": len(selected),
                        "unit_id": unit_id,
                        "proposals": len(item["anns"]),
                        "elapsed_seconds": time.time() - start,
                    }
                ),
                flush=True,
            )
    write_csv(output_dir / "cache/cache_manifest.csv", cache_rows)
    write_json(
        output_dir / "RUN_STATUS.json",
        {
            "status": "complete",
            "mode": args.mode,
            "unit_count": len(selected),
            "proposal_count": sum(int(row["proposal_count"]) for row in cache_rows),
            "elapsed_seconds": time.time() - start,
        },
    )


def native_selection(item: dict[str, Any], key: str) -> tuple[np.ndarray, float, int]:
    annotations = item["anns"]
    if not annotations:
        return np.zeros((item["H"], item["W"]), dtype=bool), float("-inf"), -1
    if key == "area":
        index = max(range(len(annotations)), key=lambda value: float(annotations[value].get("area", 0.0)))
    else:
        index = max(range(len(annotations)), key=lambda value: float(annotations[value].get(key, 0.0)))
    return annotations[index]["segmentation"].astype(bool), float(annotations[index].get(key, 0.0)), index


def write_prediction(output_dir: Path, method: str, unit_id: str, mask: np.ndarray) -> None:
    path = output_dir / "predictions" / method / f"{unit_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)


def evaluate(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    release_dir = args.release_dir.resolve()
    split_dir = args.split_dir.resolve()
    manifest, assignment_map = load_authority(release_dir, split_dir)
    expected = (
        smoke_units(manifest, assignment_map)
        if args.mode == "smoke"
        else [row["unit_id"] for row in manifest]
    )
    missing = [unit_id for unit_id in expected if not cache_path(output_dir, unit_id).is_file()]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} cache items")

    from paper_reproduction.experiments.selector_features import (
        FULL_FEATURES,
        proposal_feature_df,
        smoke_ranker_param_grid,
        train_ranker_from_params,
        tune_ranker_params,
    )

    rows: list[dict[str, Any]] = []
    native_definitions = {
        "sam_largest": "area",
        "sam_stability": "stability_score",
        "sam_predicted_iou": "predicted_iou",
    }
    for index, unit_id in enumerate(expected, start=1):
        item = load_cache(cache_path(output_dir, unit_id))
        for method, key in native_definitions.items():
            mask, score, proposal_index = native_selection(item, key)
            rows.append(score_mask(item, method, mask, score, proposal_index))
            write_prediction(output_dir, method, unit_id, mask)
        if index % 50 == 0:
            print(f"native evaluation {index}/{len(expected)}", flush=True)

    for fold in range(5):
        train_ids = [
            unit_id for unit_id in expected
            if int(assignment_map[unit_id]["evaluation_fold"]) not in {fold, (fold + 1) % 5}
        ]
        validation_ids = [
            unit_id for unit_id in expected
            if int(assignment_map[unit_id]["evaluation_fold"]) == (fold + 1) % 5
        ]
        test_ids = [
            unit_id for unit_id in expected
            if int(assignment_map[unit_id]["evaluation_fold"]) == fold
        ]
        if not test_ids:
            continue
        train_items = [load_cache(cache_path(output_dir, value)) for value in train_ids]
        validation_items = [load_cache(cache_path(output_dir, value)) for value in validation_ids]
        grid = smoke_ranker_param_grid() if args.mode == "smoke" else None
        params = tune_ranker_params(
            train_vis=train_items,
            val_vis=validation_items,
            feature_names=FULL_FEATURES,
            seed=42 + fold,
            param_grid=grid,
        )
        fit_params = {key: params[key] for key in (
            "max_depth", "learning_rate", "max_leaf_nodes", "l2_regularization", "min_samples_leaf"
        )}
        final_model = train_ranker_from_params(
            train_vis=train_items + validation_items,
            params=fit_params,
            feature_names=FULL_FEATURES,
            seed=42 + fold,
        )
        write_json(output_dir / "models" / f"fold_{fold}_learned_iou.json", params)
        del train_items, validation_items
        gc.collect()
        for unit_id in test_ids:
            item = load_cache(cache_path(output_dir, unit_id))
            features = proposal_feature_df(item)
            if features.empty or final_model is None:
                mask = np.zeros((item["H"], item["W"]), dtype=bool)
                score, proposal_index = float("-inf"), -1
            else:
                scores = final_model.predict(features[FULL_FEATURES].values)
                selected = int(np.argmax(scores))
                proposal_index = int(features.iloc[selected]["prop_idx"])
                mask = item["anns"][proposal_index]["segmentation"].astype(bool)
                score = float(scores[selected])
            rows.append(score_mask(item, "learned_iou", mask, score, proposal_index))
            write_prediction(output_dir, "learned_iou", unit_id, mask)
        print(f"learned selector fold {fold} complete", flush=True)
        del final_model
        gc.collect()

    write_csv(output_dir / "results/per_unit_metrics.csv", rows)
    summary = aggregate_rows(rows)
    write_csv(output_dir / "results/paper_table.csv", summary)
    method_counts = Counter(row["method"] for row in rows)
    expected_per_method = len(expected)
    invalid = {method: count for method, count in method_counts.items() if count != expected_per_method}
    if invalid:
        raise RuntimeError(f"Incomplete method rows: {invalid}")
    write_json(
        output_dir / "results/VALIDATION.json",
        {
            "status": "PASS",
            "mode": args.mode,
            "unit_count": len(expected),
            "methods": dict(method_counts),
            "test_unit_unique_per_method": True,
            "native_grid": True,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.command == "generate":
        generate(args)
    elif args.command == "evaluate":
        evaluate(args)


if __name__ == "__main__":
    main()

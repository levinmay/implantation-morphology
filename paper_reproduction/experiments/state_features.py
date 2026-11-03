"""Extract morphology and appearance features from fold-specific masks."""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from pipeline.features import extract_features
from paper_reproduction.experiments.study import load_authority
from pipeline.cleanup import topology
from pipeline.io import sha256_path
from paper_reproduction.experiments.validation import validate_feature_set
from paper_reproduction.experiments.run_embryo577_nnunet_benchmark import fold_paths, verify_staging_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project-root", "release-dir", "split-dir", "nnunet-dir", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--feature-set", type=Path,
                        default=Path(__file__).resolve().parents[2] / "pipeline/feature_set.json")
    parser.add_argument("--full-feature-pool", action="store_true",
                        help="Extract all 91 candidates for training-only selection")
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    parser.add_argument("--staging-mask-dir", type=Path, required=True)
    parser.add_argument("--dataset-id-base", type=int, default=986)
    args = parser.parse_args()
    feature_set = json.loads(args.feature_set.read_text())
    validate_feature_set(feature_set)
    manifest, assignments = load_authority(args.release_dir, args.split_dir)
    fold = args.outer_fold
    paths = fold_paths(args.nnunet_dir.resolve(), args.dataset_id_base + fold, fold)
    cases, verified = verify_staging_inputs(paths, fold)
    case_map = {row["unit_id"]: row for row in cases}
    if set(case_map) != {row["unit_id"] for row in manifest}:
        raise ValueError("Upstream cases differ from the feature manifest")
    mask_root = args.staging_mask_dir / f"outer_fold_{fold}"
    provenance = json.loads((mask_root / "PROVENANCE.json").read_text())
    if any(provenance.get(key) != value for key, value in verified.items()):
        raise ValueError("Staging masks do not match the verified upstream producer")
    appearance_rows, morphology_rows = [], []
    for row in manifest:
        unit = row["unit_id"]
        split = assignments[unit]
        case = case_map[unit]
        if (int(case["source_evaluation_fold"]) != int(split["evaluation_fold"])
                or str(case["component_id"]) != str(split["component_id"])):
            raise ValueError("Upstream and staging fold assignments differ")
        image_path = args.project_root / row["crop_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask_path = mask_root / "raw_predictions" / f"{case['case_id']}.png"
        if sha256_path(mask_path) != provenance["mask_sha256"][case["case_id"]]:
            raise ValueError("Staging mask changed after inference")
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise FileNotFoundError(f"Missing image or prediction for {unit}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cleaned = topology(mask > 0)
        if args.full_feature_pool:
            from paper_reproduction.experiments.candidate_features import extract_candidates
            whole, within, morphology = extract_candidates(image, cleaned)
        else:
            whole, within, morphology = extract_features(image, cleaned, feature_set)
        identity = {name: row[name] for name in ("unit_id", "parent_label", "pre_substage")}
        identity.update({name: split[name] for name in ("component_id", "evaluation_fold")})
        identity["producer_outer_fold"] = fold
        for source, features in (("whole_crop", whole), ("nnunet_250_topology", within)):
            appearance_rows.append({**identity, "mask_source": source, **features})
        morphology_rows.append({"unit_id": unit, "producer_outer_fold": fold, **morphology})
    private_dir = args.output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(appearance_rows).to_csv(private_dir / "APPEARANCE_FEATURES.csv", index=False)
    pd.DataFrame(morphology_rows).to_csv(private_dir / "MORPHOLOGY_FEATURES.csv", index=False)
    (private_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()

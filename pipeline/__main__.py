"""Command-line interface for new images, existing masks, and staging models."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "segment": "Predict embryo masks with a trained nnU-Net model",
        "characterize": "Measure supplied masks; optionally predict developmental stages",
        "run": "Predict masks, clean them and measure embryo features",
    }
    for name, description in descriptions.items():
        command = sub.add_parser(name, help=description, description=description)
        command.add_argument("--images", required=True, type=Path,
                             help="Folder of 8-bit RGB or grayscale PNG images")
        command.add_argument("--output", required=True, type=Path,
                             help="New output folder; must not already exist")
        if name == "characterize":
            command.add_argument("--masks", required=True, type=Path,
                                 help="Binary PNG masks with matching image filenames and dimensions")
        else:
            command.add_argument("--model", required=True, type=Path,
                                 help="nnU-Net folder containing plans.json, dataset.json and checkpoints")
            command.add_argument("--folds", required=True, nargs="+",
                                 help="Internal nnU-Net folds; use 0 for the paper models")
            command.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda",
                                 help="Inference device (default: cuda)")
            command.add_argument("--checkpoint", default="checkpoint_final.pth",
                                 help="Checkpoint filename within each model fold")
        if name != "segment":
            command.add_argument("--stage-model", type=Path, help="Optional fixed-feature stage classifier (.joblib); load only trusted files")
    fit = sub.add_parser("fit-stage", help="Fit a stage classifier to your labeled feature table",
                         description="Fit a classifier without cross-validation or performance evaluation")
    fit.add_argument("--features", required=True, type=Path,
                     help="CSV with the 15 pipeline features and a stage column")
    fit.add_argument("--output", required=True, type=Path,
                     help="New .joblib model file")
    fit.add_argument("--model-name", default="logistic_C1",
                     help="Classifier from pipeline.staging.candidates (default: logistic_C1)")
    args = parser.parse_args()
    if args.command == "fit-stage":
        import pandas as pd
        from .staging import fit_model, save_model
        if args.output.exists():
            raise FileExistsError(args.output)
        save_model(fit_model(pd.read_csv(args.features), args.model_name), args.output)
    elif args.command == "characterize":
        from .characterize import characterize
        characterize(args.images, args.masks, args.output, args.stage_model)
    else:
        from .segmentation import segment
        destination = args.output
        if args.command == "run":
            destination.mkdir(parents=True, exist_ok=False)
            destination = destination / "segmentation"
        masks = segment(args.images, args.model, destination, args.folds, args.device, args.checkpoint)
        if args.command == "run":
            from .characterize import characterize
            characterize(args.images, masks, args.output / "characterization", args.stage_model)
            import json
            from .io import write_checksums
            (args.output / "RUN_COMPLETE.json").write_text(json.dumps({
                "status": "complete", "segmentation": "segmentation/PROVENANCE.json",
                "characterization": "characterization/PROVENANCE.json",
            }, indent=2) + "\n")
            write_checksums(args.output)


if __name__ == "__main__":
    main()

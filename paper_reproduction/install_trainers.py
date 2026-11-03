"""Check or install the paper trainers in the active nnU-Net virtual environment."""

import argparse
import importlib
import importlib.util
import sys
from pathlib import Path

NAMES = ("nnUNetTrainer_seeded.py", "nnUNetTrainer_250epochs_seeded.py")


def install_files(source: Path, target: Path, install: bool) -> None:
    payloads = {name: (source / name).read_bytes() for name in NAMES}
    for name, content in payloads.items():
        path = target / name
        if path.is_symlink() or (path.exists() and path.read_bytes() != content):
            raise FileExistsError(f"Different trainer already exists; nothing replaced: {path}")
    missing = [name for name in NAMES if not (target / name).exists()]
    if missing and not install:
        raise FileNotFoundError(f"Missing trainers: {', '.join(missing)}; run with --install")
    for name in missing:
        with (target / name).open("xb") as handle:
            handle.write(payloads[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true", help="Add missing files; never replace a trainer")
    args = parser.parse_args()
    if args.install and sys.prefix == sys.base_prefix:
        raise RuntimeError("Activate a dedicated virtual environment before installing trainers")
    spec = importlib.util.find_spec("nnunetv2")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Install the segmentation dependencies first")
    package = Path(next(iter(spec.submodule_search_locations))).resolve()
    if args.install and not package.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("nnU-Net must be installed inside the active virtual environment")
    target = package / "training/nnUNetTrainer/variants/training_length"
    if args.install and not target.resolve().is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError("Trainer destination must stay inside the active virtual environment")
    install_files(Path(__file__).with_name("trainers"), target, args.install)
    importlib.invalidate_caches()
    for name in NAMES:
        module = importlib.import_module("nnunetv2.training.nnUNetTrainer.variants.training_length." + name[:-3])
        if not isinstance(getattr(module, name[:-3], None), type):
            raise RuntimeError(f"Trainer class is not importable: {name}")
    print("PASS: matching trainer files and importable classes")


if __name__ == "__main__":
    main()

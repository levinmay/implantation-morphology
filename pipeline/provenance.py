"""Software identity for generated measurements and prediction caches."""

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import __version__
from .io import sha256_path


def software_identity() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "scikit-image", "scikit-learn",
                 "opencv-python-headless", "Pillow", "torch", "nnunetv2"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    directory = Path(__file__).resolve().parent
    sources = sorted([*directory.glob("*.py"), directory / "feature_set.json"])
    return {"pipeline_version": __version__, "python": platform.python_version(),
            "packages": packages, "source_sha256": {p.name: sha256_path(p) for p in sources}}

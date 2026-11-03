"""Validated image/mask inputs and output checksums."""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"RGB", "L"}:
            raise ValueError(f"Expected 8-bit RGB or grayscale image: {path.name}")
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if min(value.shape[:2]) < 3:
        raise ValueError("Images must be at least 3 by 3 pixels")
    return value


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        value = np.asarray(image)
    if value.ndim != 2 or not set(np.unique(value)).issubset({0, 1, 255}):
        raise ValueError(f"Expected a binary 0/1 or 0/255 mask: {path.name}")
    return value > 0


def image_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("*.png"))
    if not files:
        raise ValueError(f"No PNG images in {directory}")
    return files


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_checksums(directory: Path) -> None:
    files = sorted(p for p in directory.rglob("*") if p.is_file()
                   and p.name != "CHECKSUMS.sha256")
    with (directory / "CHECKSUMS.sha256").open("x") as handle:
        for path in files:
            handle.write(f"{sha256_path(path)}  {path.relative_to(directory).as_posix()}\n")

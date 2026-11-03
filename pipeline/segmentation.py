"""Run a compatible exported nnU-Net model on new RGB PNG images."""

import json
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from .io import image_files, read_image, read_mask, sha256_path, write_checksums
from .provenance import software_identity


def segment(images: Path, model: Path, output: Path, folds: list[str],
            device: str = "cuda", checkpoint: str = "checkpoint_final.pth") -> Path:
    images, model, output = Path(images), Path(model), Path(output)
    if device not in {"cuda", "cpu", "mps"}:
        raise ValueError("Device must be cuda, cpu, or mps")
    if not folds or any(f != "all" and not str(f).isdigit() for f in folds):
        raise ValueError("Specify model folds as nonnegative integers or all")
    if Path(checkpoint).name != checkpoint:
        raise ValueError("Checkpoint must be a filename within each model fold")
    files = image_files(images)
    dataset = json.loads((model / "dataset.json").read_text())
    if dataset.get("file_ending") != ".png" or len(dataset.get("channel_names", {})) != 3:
        raise ValueError("This adapter requires a three-channel RGB PNG model")
    if set(dataset.get("labels", {}).values()) != {0, 1}:
        raise ValueError("This adapter requires a binary embryo segmentation model")
    assets = [model / "plans.json", model / "dataset.json"]
    assets += [model / f"fold_{f}" / checkpoint for f in folds]
    hashes = {p.relative_to(model).as_posix(): sha256_path(p) for p in assets}
    executable = shutil.which("nnUNetv2_predict_from_modelfolder")
    if executable is None:
        raise RuntimeError("Install the segmentation dependencies in the active environment")
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "nnunet_input"
    inputs.mkdir()
    mapping = []
    for index, path in enumerate(files):
        image = read_image(path)
        case = f"image_{index:06d}"
        Image.fromarray(image).save(inputs / f"{case}_0000.png")
        mapping.append({"image": path.name, "case": case, "shape": list(image.shape[:2]),
                        "sha256": sha256_path(path)})
    predictions = output / "nnunet_predictions"
    command = [executable, "-i", str(inputs), "-o", str(predictions), "-m", str(model),
               "-f", *map(str, folds), "-chk", checkpoint, "-device", device,
               "-npp", "1", "-nps", "1"]
    subprocess.run(command, check=True)
    if {p.relative_to(model).as_posix(): sha256_path(p) for p in assets} != hashes:
        raise ValueError("Model files changed during inference")
    expected = {f"{r['case']}.png" for r in mapping}
    if {p.name for p in predictions.glob("*.png")} != expected:
        raise RuntimeError("nnU-Net did not produce the expected masks")
    masks = output / "masks"
    masks.mkdir()
    for row in mapping:
        mask = read_mask(predictions / f"{row['case']}.png")
        if list(mask.shape) != row["shape"]:
            raise RuntimeError("nnU-Net prediction changed the image grid")
        Image.fromarray(mask.astype("uint8") * 255).save(masks / row["image"])
    (output / "PROVENANCE.json").write_text(json.dumps({
        "schema_version": 2, "inputs": mapping, "model_sha256": hashes,
        "software": software_identity(),
        "folds": folds, "device": device, "checkpoint": checkpoint,
    }, indent=2) + "\n")
    write_checksums(output)
    return masks

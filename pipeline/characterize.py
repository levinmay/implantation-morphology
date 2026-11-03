"""Characterize paired images and predicted masks without cohort assumptions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .cleanup import topology, HOLE_PARAMETERS
from .features import default_feature_set, extract_features
from .io import image_files, read_image, read_mask, sha256_path, write_checksums
from .tcmi import tcmi_score
from .measurements import mask_measurements
from .provenance import software_identity


def characterize(images: Path, masks: Path, output: Path, stage_model: Path | None = None) -> Path:
    images, masks, output = Path(images), Path(masks), Path(output)
    files = image_files(images)
    if {p.name for p in files} != {p.name for p in masks.glob("*.png")}:
        raise ValueError("Images and masks must have exactly matching PNG filenames")
    model_hash = sha256_path(Path(stage_model)) if stage_model else None
    model = None
    if stage_model is not None:
        from .staging import load_model
        model = load_model(stage_model)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cleaned_masks").mkdir()
    feature_set = default_feature_set()
    rows, provenance = [], []
    for path in files:
        image, raw = read_image(path), read_mask(masks / path.name)
        if image.shape[:2] != raw.shape:
            raise ValueError(f"Image/mask grid mismatch: {path.name}")
        cleaned = topology(raw)
        Image.fromarray(cleaned.astype(np.uint8) * 255).save(output / "cleaned_masks" / path.name)
        row = {"image": path.name, "status": "ok" if cleaned.any() else "empty_mask",
               "raw_area_px2": int(raw.sum()), "cleaned_area_px2": int(cleaned.sum()),
               "tcmi_raw": tcmi_score(raw)}
        row.update(mask_measurements(cleaned))
        row.update({name: np.nan for name in feature_set["combined"]})
        if cleaned.any():
            _, within, morphology = extract_features(image, cleaned, feature_set, include_whole=False)
            row.update({**morphology, **within})
        rows.append(row)
        provenance.append({"image": path.name, "image_sha256": sha256_path(path),
                           "mask_sha256": sha256_path(masks / path.name)})
    frame = pd.DataFrame(rows)
    if model is not None:
        from .staging import predict_stages, CLASSES
        valid = frame.status == "ok"
        frame["stage"] = "not_assessed_empty_mask"
        for label in CLASSES:
            frame[f"prob_{label}"] = np.nan
        if valid.any():
            prediction = predict_stages(model, frame.loc[valid])
            for column in prediction:
                frame.loc[valid, column] = prediction[column].to_numpy()
    frame.to_csv(output / "measurements.csv", index=False)
    (output / "PROVENANCE.json").write_text(json.dumps({
        "schema_version": 3, "inputs": provenance, "software": software_identity(),
        "feature_set": feature_set, "cleanup": "largest_component_then_tiny_holes",
        "cleanup_parameters": {"connectivity": 8, **HOLE_PARAMETERS},
        "tcmi_source": "raw_mask", "staging_source": "cleaned_mask",
        "stage_model_sha256": model_hash,
    }, indent=2) + "\n")
    if stage_model and sha256_path(Path(stage_model)) != model_hash:
        raise ValueError("Staging model changed during characterization")
    write_checksums(output)
    return output

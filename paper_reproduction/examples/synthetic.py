"""Generate artificial image/mask pairs for the CPU quickstart."""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image


def generate(output: Path, training_example: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    (output / "masks").mkdir()
    y, x = np.ogrid[:96, :128]
    rng = np.random.default_rng(42)
    disk = (y - 48) ** 2 + (x - 58) ** 2 < 25 ** 2
    lobed = disk | ((y - 40) ** 2 + (x - 85) ** 2 < 14 ** 2)
    noisy = disk.copy()
    noisy[2:5, 2:5] = True
    noisy[42:44, 53:55] = False
    for name, mask in (("round", disk), ("two_lobe", lobed), ("noisy", noisy),
                       ("empty", np.zeros_like(disk))):
        intensity = np.clip(170 - 60 * mask + rng.normal(0, 7, mask.shape), 0, 255).astype("uint8")
        image = np.repeat(intensity[..., None], 3, axis=2)
        Image.fromarray(image).save(output / "images" / f"{name}.png")
        Image.fromarray(mask.astype("uint8") * 255).save(output / "masks" / f"{name}.png")
    if training_example:
        import pandas as pd
        from pipeline.features import default_feature_set, extract_features
        from pipeline.staging import CLASSES
        rows = []
        for index in range(20):
            mask = lobed if index % 2 else disk
            intensity = np.clip(170 - 60 * mask + rng.normal(0, 7, mask.shape), 0, 255).astype("uint8")
            image = np.repeat(intensity[..., None], 3, axis=2)
            _, appearance, morphology = extract_features(image, mask, default_feature_set())
            # Arbitrary labels exercise the CSV contract, not developmental biology.
            rows.append({**morphology, **appearance, "stage": CLASSES[index % 4]})
        pd.DataFrame(rows).to_csv(output / "format-only-training.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-example", action="store_true")
    args = parser.parse_args()
    generate(args.output, args.training_example)

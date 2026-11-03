# Whole-embryo segmentation and characterization

This package segments whole embryos in brightfield microscopy images, cleans
the masks, and measures embryo shape and appearance. A classifier predicts four
developmental states: early pre-implantation/hatching,
floating/apposition/adhesion, implantation or later, and degeneration.
The Two-Circle Morphology Index (TCMI) describes visible hatching-associated
two-lobe shape. This is research software, not a clinical predictor.

## Install

Use Python 3.11 on Linux. Run these commands from the directory containing
this README. The environment and outputs are saved outside that directory.

```bash
python3.11 -m venv ../embryo-env
source ../embryo-env/bin/activate
python -m pip install -e .
```

## Try the supplied example

This example measures six images of one embryo using supplied masks, without
running segmentation or staging.

```bash
python -m pipeline characterize \
  --images paper_reproduction/examples/figure1/images \
  --masks paper_reproduction/examples/figure1/predicted_masks \
  --output ../figure1-analysis
```

Open `../figure1-analysis/measurements.csv` for the measurements and
`../figure1-analysis/cleaned_masks/` for the masks. To check these against the
supplied results:

```bash
python -m paper_reproduction.examples.verify_figure1 --output ../figure1-analysis
```

Use a new output directory for each run. For command options, run
`python -m pipeline --help` or `python -m pipeline characterize --help`.
See [example details](paper_reproduction/examples/README.md) for the data source
and a synthetic classifier-fitting example.

## Run on your own images

Put your images in a folder such as `../images/`, with one embryo per image.
Use 8-bit RGB or grayscale PNG files with a lowercase `.png` extension. Convert
16-bit microscopy images using an appropriate, documented intensity range first.

If you already have masks, use:

```bash
python -m pipeline characterize \
  --images ../images --masks ../masks --output ../embryo-measurements
```

Masks must have the same filenames and dimensions as their images, with
background 0 and foreground 1 or 255. To generate masks and predict stages,
follow the model setup below.

## Run with trained models

Model weights are not included.
See [paper reproduction](paper_reproduction/RUNNING.md) for training instructions.

The example paths below assume models are placed in `../models/`.

The example uses the segmenter and stage classifier from outer fold 1, which
also produced the Figure 1 predictions. Use models from the same outer fold
and keep `--folds 0`, the internal nnU-Net fold.
See [model files](pipeline/MODELS.md) for the required layout.

Install the segmentation dependencies and matching nnU-Net trainer classes:

```bash
python -m pip install -e '.[segmentation]'
python -m paper_reproduction.install_trainers --install
python -m paper_reproduction.install_trainers
```

Then run segmentation, mask cleanup, measurements, and stage prediction:

```bash
python -m pipeline run \
  --images ../images \
  --model ../models/outer_fold_1/segmentation --folds 0 \
  --stage-model ../models/outer_fold_1/stage_model.joblib \
  --device cpu --output ../embryo-results
```

Use `--device cuda` for a compatible GPU. To run the Figure 1 images instead,
set `--images paper_reproduction/examples/figure1/images`. Fresh predictions
may differ from saved masks and are not checked by `verify_figure1`.
Load only trusted models.

## Outputs

The combined run writes these files inside `../embryo-results/`:

- `segmentation/masks/`: predicted masks at the original image dimensions.
- `characterization/cleaned_masks/`: masks after cleanup.
- `characterization/measurements.csv`: one row per image, with morphology,
  appearance, TCMI, and the predicted stage and class probabilities.

Empty masks are flagged rather than assigned a stage. The pipeline reports area
in pixels squared. Physical-unit figures in the paper used study-specific scales
applied separately.
See [output fields](pipeline/OUTPUTS.md).
`RUN_COMPLETE.json` and `CHECKSUMS.sha256` mark a completed combined run.

## Paper

May Levin, Molika Sinha, Pietro Liò and Matteo Molè. *Whole-Embryo Segmentation
and Developmental Characterization During In Vitro Implantation.*
Accepted at ML4H 2026, PMLR volume 346. [BibTeX](CITATION.bib).

## Further instructions

- [Pipeline usage](pipeline/README.md): individual commands and classifier fitting.
- [Output fields](pipeline/OUTPUTS.md): measurements, units, and stage labels.
- [Paper reproduction](paper_reproduction/README.md): training and evaluation protocols.

Code: MIT. Figure 1 example data: CC BY 4.0. Other study
data, figure-generation code, and manuscript files are not included.

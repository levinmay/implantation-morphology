# Pipeline usage

Run commands from the repository root. After installation, you can use `embryo-pipeline`
instead of `python -m pipeline`.
Use 8-bit RGB or grayscale PNGs with one embryo per image. For 16-bit images,
choose and document the intensity range used for conversion.
Masks must contain 0/1 or 0/255 values and match their images' dimensions and
filenames. Saved masks keep the input dimensions, which can vary between images.
Appearance and TCMI use normalized, resized copies.

## Existing masks

```bash
python -m pipeline characterize --images images --masks masks --output analysis
```

Outputs: `cleaned_masks/`, `measurements.csv`, `PROVENANCE.json`, and checksums.
See [output definitions](OUTPUTS.md) for details.
Cleanup keeps the largest connected component and fills tiny holes. It leaves
large cavities and does not make masks circular. Area is in pixels squared.
The 15 staging descriptors use cleaned masks and TCMI uses raw masks. Empty
masks have no staging descriptors or predicted stage. Review masks before
interpreting measurements.

A completed output has checksums in its top-level directory. A combined `run`
also has `RUN_COMPLETE.json`. Failed runs can leave partial files, so retry in
a new directory.

## Segmentation

```bash
python -m pipeline segment --images images --model model --folds 0 --output segmentation
python -m pipeline run --images images --model model --folds 0 --output complete-run
```

`run` segments images, then measures the masks. Segmentation uses nnU-Net's
[model-folder inference interface](https://github.com/MIC-DKFZ/nnUNet/blob/v2.6.4/nnunetv2/inference/predict_from_raw_data.py).
Supply the binary RGB PNG model, plans, checkpoints, and trainer listed in
[model files](MODELS.md).
Use `--folds 0` for each paper segmenter and pair it with the stage classifier
from the same outer fold. Use `--device cpu` when appropriate for your resources.
The run keeps its converted RGB inputs and leaves the original files untouched.

## Fit a classifier on your own labeled data

Add a `stage` column to the 15 feature columns from `characterize`, using the
four developmental labels in `staging.CLASSES`. These are not embryo-quality labels.

```bash
python -m pipeline fit-stage --features labeled-training.csv --output classifier.joblib
python -m pipeline characterize --images images --masks masks --output staged \
  --stage-model classifier.joblib
```

`fit-stage` fits one classifier, using balanced logistic regression
(`logistic_C1`) by default. It does not run the paper's model selection or
cross-validation. Use `--model-name` to choose another model from
`staging.candidates`. Keep evaluation embryos out of fitting and tuning.
Follow the [reproduction protocol](../paper_reproduction/RUNNING.md) to repeat the paper's comparisons.

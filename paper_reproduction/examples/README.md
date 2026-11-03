# Examples

## Figure 1 embryo

`figure1/` contains six cropped RGB images at DPF 5–10 with their original
dimensions and intensities. Filenames are anonymous and embedded metadata has
been removed. Masks are binary PNGs (0 background, 255 embryo).

- `images/`: images used for analysis, before resizing for the figure.
- `reference_masks/`: manually edited study masks.
- `predicted_masks/`: saved seed-42 out-of-fold nnU-Net predictions.
- `cleaned_masks/`: predictions after mask cleanup.
- `manifest.csv`: DPF and dimensions.
- `expected_measurements.csv`: measurements calculated from the predictions.

The [main guide](../../README.md) covers mask verification and inference with
separately supplied models. This study embryo is not an independent validation
set. Clinical outcomes, original IDs, source coordinates, and physical
calibration are omitted. Sizes are in pixels.
Example data: [CC BY 4.0](figure1/DATA_LICENSE.md). Code: MIT.

## Synthetic inputs

From the repository root:

```bash
python -m paper_reproduction.examples.synthetic --output demo-inputs --training-example
python -m pipeline characterize --images demo-inputs/images \
  --masks demo-inputs/masks --output demo-results
```

This creates four artificial cases: round, two-lobed, fragmented, and empty.
Outputs are in `demo-results/measurements.csv` and `cleaned_masks/`. TCMI uses
raw masks and staging uses cleaned masks. Empty masks receive no stage and do
not confirm embryo absence. The example tests execution, not accuracy or biology.

To test classifier fitting and loading:

```bash
python -m pipeline fit-stage --features demo-inputs/format-only-training.csv \
  --output format-only-classifier.joblib
python -m pipeline characterize --images demo-inputs/images --masks demo-inputs/masks \
  --stage-model format-only-classifier.joblib --output demo-staging
```

The CSV has 20 synthetic rows with arbitrary stage labels. Use it to test
fitting and loading, not to assess biology or performance. Actual training
requires reviewed stage labels.

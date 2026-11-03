# Reproduce the paper

These scripts cover the 577-image study: segmentation, fixed-feature and
training-only classification, TCMI, and calibration sensitivity. They use the
paper's five grouped folds and require private study inputs. For your own labeled data, see
[classifier fitting](../pipeline/README.md#fit-a-classifier-on-your-own-labeled-data).

- [Running the analyses](RUNNING.md): training and evaluation commands.
- [Input formats](MANIFEST_SCHEMA.md): required tables and labels.
- `experiments/`: training, feature extraction, evaluation, and statistics.
- `configs/`: recorded experiment settings.
- `folds/`: anonymized fold assignments.
- [Results](results/README.md): saved aggregate tables for the paper.
- `trainers/`: seeded nnU-Net trainer classes.
- [Examples](examples/README.md): the Figure 1 sequence and synthetic inputs.

Only the Figure 1 example images and masks are included. Other study data,
per-image results, weights, figure scripts, and manuscript sources are omitted.
See [model files](../pipeline/MODELS.md) for model requirements.

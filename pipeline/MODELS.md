# Model files

Pretrained weights are not included. Use compatible trained models with the
commands in the [main README](../README.md). Training and evaluation commands
are in [paper reproduction](../paper_reproduction/RUNNING.md).

For segmentation, supply an nnU-Net directory containing:

```text
segmentation/
  dataset.json
  plans.json
  fold_0/checkpoint_final.pth
```

The model must use binary embryo labels and three-channel RGB PNG inputs.
Install its matching trainer class before inference.

Each paper fold has a separate segmenter and stage classifier. Keep them paired;
the stage classifier was fitted using features from that segmenter's masks.
Each segmenter uses internal fold 0, so specify `--folds 0`.
The Figure 1 sequence uses pair 1. These are fold-specific models, not a model
trained on the whole cohort or a validated ensemble.

The pipeline accepts the fixed-feature classifiers from the reproduction run
or a classifier saved by `fit-stage`. The training-only selection models use
different features in each fold and cannot be passed to `--stage-model`.

A pipeline stage-model file stores the fitted classifier, feature order, class
labels, and scikit-learn version together. See [pipeline usage](README.md#fit-a-classifier-on-your-own-labeled-data)
to fit and save one. Keep model files outside the repository.

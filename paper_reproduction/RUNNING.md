# Running the analyses

Run commands from the repository root with your Python environment activated.
See [MANIFEST_SCHEMA.md](MANIFEST_SCHEMA.md) for input formats. Keep private
study data and per-image outputs in your approved research environment.

Install the full dependencies in a dedicated Python 3.11 virtual environment,
then add and check the paper trainers:

```bash
python -m pip install -e '.[segmentation,sam]'
python -m paper_reproduction.install_trainers --install
python -m paper_reproduction.install_trainers
```

The installer adds missing trainers and stops if an existing trainer differs.
The last command checks the installation without changing it.

`paper_reproduction/configs/benchmark.json` records the experiment settings.
The scripts use command-line arguments rather than loading this file. Run
each script with `--help` to see its options.

## Segmentation

Set `PROJECT_DIR` to the root for manifest `crop_path` entries, `MASK_RELEASE` to
the directory containing `tables/embryo_mask_manifest.csv`, and `SPLIT_DIR` to
the directory containing `unit_fold_assignment.csv`. Set `SAM_CHECKPOINT` to
the authorized ViT-H checkpoint and verify its SHA-256 against
`paper_reproduction/configs/benchmark.json`. Set the output variables below to separate new
directories. Reuse each method's directory across its successive commands.

```bash
python -m paper_reproduction.experiments.run_embryo577_sam_benchmark generate \
  --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
  --split-dir "$SPLIT_DIR" --output-dir "$SAM_OUTPUT" \
  --sam-checkpoint "$SAM_CHECKPOINT" --mode full
python -m paper_reproduction.experiments.run_embryo577_sam_benchmark evaluate \
  --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
  --split-dir "$SPLIT_DIR" --output-dir "$SAM_OUTPUT" --mode full

python -m paper_reproduction.experiments.run_embryo577_nnunet_benchmark prepare \
  --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
  --split-dir "$SPLIT_DIR" --output-dir "$NNUNET_OUTPUT"
```

After preparation, train and predict all five folds on a GPU. The loop stops
if a fold fails.

```bash
(
  set -e
  for fold in 0 1 2 3 4; do
    python -m paper_reproduction.experiments.run_embryo577_nnunet_benchmark run-fold \
      --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
      --split-dir "$SPLIT_DIR" --output-dir "$NNUNET_OUTPUT" \
      --fold "$fold" --device cuda
  done
)
```

When all five folds finish, generate the summary tables:

```bash
python -m paper_reproduction.experiments.run_embryo577_nnunet_benchmark aggregate \
  --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
  --split-dir "$SPLIT_DIR" --output-dir "$NNUNET_OUTPUT"
python -m paper_reproduction.experiments.analyze_embryo577_segmentation \
  --sam-metrics "$SAM_OUTPUT/results/per_unit_metrics.csv" \
  --nnunet-metrics "$NNUNET_OUTPUT/results/nnunet_per_unit_metrics.csv" \
  --output-dir "$SEGMENTATION_SUMMARIES" --bootstrap-repetitions 5000
```

The main segmentation results use raw predictions. The runner also evaluates
mask cleanup, which is the same for every stage. These commands use seed 42
and 250 epochs. For the three-seed comparison, repeat preparation and training
in separate directories with `--seed 43` and `--seed 44` and report all three
seeds. Preparation regenerates nnU-Net plans, so check them against the
recorded configuration when reproducing the paper.

## Developmental-state classification with fixed features

For each outer fold, use its seed-42 segmenter to predict all images. The
command keeps the original test predictions and predicts the remaining images
without retraining. Start with new `STAGING_MASKS` and `FEATURE_OUTPUT` directories.

```bash
(
  set -e
  for fold in 0 1 2 3 4; do
    python -m paper_reproduction.experiments.run_embryo577_nnunet_benchmark predict-staging \
      --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
      --split-dir "$SPLIT_DIR" --output-dir "$NNUNET_OUTPUT" \
      --fold "$fold" --staging-mask-dir "$STAGING_MASKS" --device cuda
    python -m paper_reproduction.experiments.state_features --project-root "$PROJECT_DIR" \
      --release-dir "$MASK_RELEASE" --split-dir "$SPLIT_DIR" \
      --nnunet-dir "$NNUNET_OUTPUT" --outer-fold "$fold" \
      --staging-mask-dir "$STAGING_MASKS" \
      --output-dir "$FEATURE_OUTPUT/outer_fold_$fold"
  done
)
python -m paper_reproduction.experiments.run_embryo577_compact_matched_ablation \
  --outer-feature-dir "$FEATURE_OUTPUT" \
  --feature-set pipeline/feature_set.json --output-dir "$STAGING_OUTPUT"
python -m paper_reproduction.experiments.analyze_embryo577_compact_staging \
  --predictions-dir "$STAGING_OUTPUT/private" --output-dir "$STAGING_SUMMARIES"
```

The feature lists and column order are fixed. Each method saves five classifiers
in `private/models/<method>/` with a completion marker and checksums, without
overwriting existing models. Use the combined 15-feature models with
`pipeline --stage-model`. The others are for ablation comparisons.

Keep the supplied folds and input-row order. Generate each fold's feature table
with its own segmenter, which has seen the training images but not the test
images. The scripts check folds, planning inputs, checkpoints, and masks.

These commands use the fixed feature sets and reproduce the combined AUROC
of 0.829. The training-only selection below reproduces the separate 0.814
result. Three whole-image boundary descriptors are unavailable and replaced
with constants in the fixed-feature comparison, as described in the appendix.

## TCMI

Set `TIMING_LABELS` to the reviewed 211-row CSV described in
`MANIFEST_SCHEMA.md`, and choose new `TCMI_OUTPUT` and `TCMI_SUMMARIES`
directories. TCMI is calculated from reference masks and raw out-of-fold
predictions. Timing labels are used only for evaluation:

```bash
python -m paper_reproduction.experiments.score_tcmi_batch --release-dir "$MASK_RELEASE" \
  --split-dir "$SPLIT_DIR" --nnunet-dir "$NNUNET_OUTPUT" \
  --timing-labels "$TIMING_LABELS" --output-dir "$TCMI_OUTPUT"
python -m paper_reproduction.experiments.analyze_tcmi \
  --scores "$TCMI_OUTPUT/private/TCMI_SCORES.csv" \
  --output-dir "$TCMI_SUMMARIES" --bootstrap-repetitions 5000
```

For a single mask, call `tcmi_score` in `pipeline/tcmi.py` with a two-dimensional
binary NumPy array. Use raw predictions for TCMI and cleaned predictions for
staging. 

## Training-only feature selection

Repeat the `state_features` command above for each fold with
`--full-feature-pool` and a new `CANDIDATE_OUTPUT` directory. Keep the same
`STAGING_MASKS` and seed-42 segmenters. This extracts 49 morphology and 42
appearance candidates using the feature definitions from the selection analysis.

```bash
(
  set -e
  for fold in 0 1 2 3 4; do
    python -m paper_reproduction.experiments.state_features \
      --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
      --split-dir "$SPLIT_DIR" --nnunet-dir "$NNUNET_OUTPUT" \
      --staging-mask-dir "$STAGING_MASKS" --outer-fold "$fold" \
      --full-feature-pool --output-dir "$CANDIDATE_OUTPUT/outer_fold_$fold"
  done
)
python -m paper_reproduction.experiments.run_training_only_selection \
  --outer-feature-dir "$CANDIDATE_OUTPUT" \
  --fixed-predictions-dir "$STAGING_OUTPUT/private" \
  --output-dir "$SELECTION_OUTPUT"
```

For each outer fold, ranking uses only the three training folds. The next
fold selects the classifier; the outer test fold is evaluated after refitting
on training plus validation. Budgets are 6 morphology, 9 appearance or 15
combined features. The combined selection is not constrained to six morphology
and nine appearance features. The ranker uses 500 Extra Trees and 10
permutations per feature. Confidence intervals use 5,000 paired bootstrap
replicates over linkage groups.

`aggregate/` contains metrics, confusion matrices, selected features and model
choices. `private/` contains predictions and fitted models. These models use
fold-specific features and are not compatible with the fixed 15-feature
`pipeline --stage-model` interface. Upstream masks for segmenter-training images
remain in-sample; this analysis changes feature selection, not segmentation.

## Conventional U-Net

Use the same manifest, fold assignments and completed nnU-Net run. Choose a
new `UNET_OUTPUT` directory. Training requires a CUDA GPU with bfloat16 support.

```bash
python -m paper_reproduction.experiments.plain_unet.run \
  --project-root "$PROJECT_DIR" --release-dir "$MASK_RELEASE" \
  --split-dir "$SPLIT_DIR" --nnunet-dir "$NNUNET_OUTPUT" \
  --output-dir "$UNET_OUTPUT"
```

This trains all five folds for 250 epochs with batch size 4. It preserves the
paper's architecture, preprocessing, augmentation, optimizer and validation
checkpoint selection. Predictions are restored to the original image grid
without mask cleanup. Aggregate files report Dice by developmental group and
the paired comparison with nnU-Net. Checkpoints and per-image results stay in
`private/`.

## Calibration sensitivity

Supply the study's private calibration table described in `MANIFEST_SCHEMA.md`.
The analysis reuses existing scales and compares all 577 images with the 520
remaining after excluding 57 appearance-scaled experiment-25 images.

```bash
python -m paper_reproduction.experiments.calibration_sensitivity \
  --input "$CALIBRATION_TABLE" --output-dir "$CALIBRATION_OUTPUT"
```

The output contains image-weighted mean and median areas by day and stage,
cohort counts, and the direction of the reported trends. It does not establish
calibration accuracy.

## Trained-model paths

If you trained with the reproduction commands, use these output paths for
outer fold 1 and the default dataset IDs:

```bash
  --model "$NNUNET_OUTPUT/folds/fold_1/nnunet_env/results/Dataset987_embryo577_fold1/nnUNetTrainer_250epochs_seeded__nnUNetPlans__2d" \
  --stage-model "$STAGING_OUTPUT/private/models/compact_morphology_plus_appearance/outer_fold_1.joblib"
```

`NNUNET_OUTPUT` and `STAGING_OUTPUT` are the training output directories set in
the reproduction instructions. For another outer fold, change both paths to
that fold. The dataset ID is `986 + fold` unless you changed `--dataset-id-base`.
Keep `--folds 0` for every outer fold.


## Local checks

The example check in the main README verifies the supplied masks and
measurements. With segmentation dependencies installed, run the tests for
feature-selection independence, preprocessing, loss and U-Net gradients:

```bash
python -m unittest paper_reproduction.experiments.tests.test_feature_selection \
  paper_reproduction.experiments.tests.test_candidate_features \
  paper_reproduction.experiments.tests.test_calibration \
  paper_reproduction.experiments.plain_unet.test_model \
  paper_reproduction.experiments.plain_unet.test_unet
```

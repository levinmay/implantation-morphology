# Paper input schema

The private microscopy release is not distributed with this repository. The
segmentation runners expect the first two private CSV files below. Timing and
physical-scale metadata are additional inputs for the corresponding analyses.

## Mask manifest

`tables/embryo_mask_manifest.csv` contains one row per embryo-by-day unit:

| Field | Meaning |
|---|---|
| `unit_id` | Unique observation identifier |
| `embryo_id` | Embryo identifier used to keep repeated observations together |
| `experiment` | Acquisition-batch label |
| `dpf` | Integer day post-fertilization |
| `parent_label` | `pre_implantation`, `implantation_or_later`, or `degeneration` |
| `pre_substage` | For pre-implantation units: `early_pre_implantation`, `hatching`, or `floating_apposition_adhesion` |
| `mask_path` | Release-relative path to the reviewed binary reference mask |
| `crop_path` | Project-relative path to the one-embryo RGB crop |

Every reference mask used by the paper was manually drawn, pixel-edited, and
reviewed.

## Fold assignment

`unit_fold_assignment.csv` contains:

| Field | Meaning |
|---|---|
| `unit_id` | Joins one-to-one with the mask manifest |
| `component_id` | Private grouping token used for leakage-safe evaluation and paired resampling |
| `evaluation_fold` | Integer fold in `0`--`4` |

For outer test fold $k$, fold $(k+1)\bmod5$ is validation and the remaining
three folds are training.

The repository publishes the exact assignment as
`folds/opaque_fold_assignments.csv` with three privacy-safe fields:

| Field | Meaning |
|---|---|
| `case_token` | Non-reversible token for one evaluated unit |
| `evaluation_fold` | Integer fold in `0`--`4` |
| `component_token` | Non-reversible grouping token preserving the leakage-safe grouping structure |

The mapping from these tokens to governed identifiers remains private. The repository includes six example images and their reference, predicted and
cleaned masks. The remaining study images and masks, identifying information
and clinical metadata are not included.

## Developmental-state feature cache

For each outer fold, the state-classification scripts consume a private feature cache with one row
per unit and mask source. It contains `unit_id`, `component_id`,
`evaluation_fold`, `producer_outer_fold`, `parent_label`, `pre_substage`, `mask_source`, and the fixed
descriptor columns in `pipeline/feature_set.json`.

Each `outer_fold_k/private/APPEARANCE_FEATURES.csv` has 1,154 rows: one `whole_crop` row and one
`nnunet_250_topology` row per unit. Both contain the identical nine appearance
column names. The accompanying `MORPHOLOGY_FEATURES.csv` has 577 unique
`unit_id` rows, `producer_outer_fold`, and six `rich_` morphology columns.
`experiments/state_features.py` creates both caches from the manifest and the complete
`outer_fold_k/raw_predictions/` set produced by `predict-staging`, applying
the fixed cleanup once. Its private `PROVENANCE.json` records producer, split,
checkpoint and mask hashes. `--outer-feature-dir` must contain all five outer
caches. Each uses segmenter k for all 577 rows, not each row's own test-fold
segmenter. Group/fold labels, producers and four-group counts are checked
before fitting; whole-image and within-mask identities must match.

## TCMI timing-score input

`python -m paper_reproduction.experiments.score_tcmi_batch --timing-labels TIMING.csv` accepts a private CSV with
exactly 211 unique eligible `unit_id` values and two binary columns:
`hatching_window_target` (62 positives) and `exact_hatching_day_target`
(60 positives). Exact-day positives are also hatching-window positives. All
identifiers must occur in the mask manifest. The custodian supplies the frozen
eligibility and timing review; the scorer does not derive them from the
developmental groups, DPF alone, or the mask shape.

The scorer obtains `component_id` and the held-out fold from the original split
assignment. It reads reference masks using manifest `mask_path` and raw nnU-Net
masks from `folds/fold_k/raw_predictions/unit_id.png`. It requires matching
grids and nonempty reference masks; an empty prediction receives TCMI zero.
It writes `private/TCMI_SCORES.csv` and `private/PROVENANCE.json` in a new run
directory. Neither file is part of the public release.

`python -m paper_reproduction.experiments.analyze_tcmi --scores SCORES.csv --output-dir OUTPUT` expects the
already eligible 211 observations for each of `reference` and `nnunet_raw`.
Required columns are `unit_id`, `component_id`, `source`, `tcmi_score`,
`hatching_window_target`, and `exact_hatching_day_target`. The last two are
binary 0/1 labels supplied by the authorized timing review, not model inputs.
The paper subset excludes five hatching-window observations whose review notes
explicitly state that hatching was not visible. These eligibility decisions
must be supplied by the custodian, not inferred from TCMI values.

AUROC uses all 211 eligible observations per source (62 hatching-window
positives; 60 exact-day positives). In `results/tcmi_aggregate.csv`, the mean,
median, quartiles, zero fraction, and above-50/above-70 fractions instead
summarize the target-positive observations. The latter fractions use strict
greater-than comparisons; they are descriptive, not selected thresholds.

## Physical calibration for biological figures

Reproducing physical-area and average-mask summaries requires the private
per-image micrometres-per-pixel scales and their provenance. Shared source
images use the same scale. Some scales are stage-informed morphology estimates;
these exploratory summaries are not independent measurements of growth.

`results/morphology_means.csv` reports image-level means, sample standard
deviations, and counts by day and developmental group, without embryo-level
weighting. Area is already in thousands of square micrometres; do not recalibrate
it. Standard deviations are not confidence intervals. `Predictive model` uses
cleaned out-of-fold masks for area and circularity, and raw masks for TCMI.
References are manually edited masks.
`early_preimplantation_hatching` is an alias of `early_pre_implantation_hatching`.

Average-mask construction is described in the paper's appendix. Figure rendering
is maintained separately; the underlying private inputs are not included.


## Full candidate feature cache

`state_features --full-feature-pool` uses the same identities, producer fields
and file layout as the compact cache. It writes 42 `texture_` columns per
appearance row and 49 `rich_` columns per morphology row. Generate these in a
separate directory from the compact tables. Missing boundary contrasts remain
missing and are handled by training-only eligibility checks and imputation.

## Calibration table

`calibration_sensitivity --input` accepts the private 577-row study table with
`unit_id`, `embryo_id`, `source_sha256`, `experiment_num`, `dpf`, `four_group`,
`calibration_basis`, `reference_area_um2` and `prediction_area_um2`.
`experiment_num` and `dpf` are numeric. The appearance-based scale category is
`experiment_dpf_morphology_normalized_median`. The script checks the full cohort
and the 57-image exclusion before calculating summaries. These inputs and the
output subset-membership table remain private.

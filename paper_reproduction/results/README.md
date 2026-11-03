# Saved results

The fixed-feature and training-only classification tables describe separate
evaluations. Keep both when comparing the code with the paper.

| Files | Evaluation |
|---|---|
| `staging_results.csv` and existing `staging_*` tables | Fixed features; combined macro AUROC 0.829 |
| `staging_training_only_*` | Training-only selection; combined macro AUROC 0.814, including the appendix confusion matrix |
| `segmentation_results.csv` | nnU-Net and SAM comparisons; nnU-Net mean Dice 0.880 |
| `plain_unet_*` | Conventional U-Net comparison; mean Dice 0.813 |
| `segmentation_mean_median.csv` | Mean and median Dice for nnU-Net and conventional U-Net |
| `calibration_*` | Full cohort and the 57-image exclusion used in the appendix |
| `tcmi_*`, `morphology_means.csv`, `topology_standardization.csv` | TCMI, physical-area summaries and mask cleanup |

`camera_ready_sources.json` records the source experiment files and checksums
for the added tables. The selection and U-Net tables were copied from completed
runs. Calibration tables retain only the two subsets reported in the paper.
No new training run was used to generate these release tables.

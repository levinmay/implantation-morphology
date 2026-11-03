# Output fields

`measurements.csv` has one row per PNG, in filename order. Empty fields are
missing values, not zero. Saved masks retain their input grid. These outputs
do not estimate embryo quality, implantation probability, or clinical outcome.

| Column | Meaning |
|---|---|
| `image` | Original PNG filename |
| `status` | `ok` means nonempty mask, not successful biological validation; `empty_mask` means no foreground |
| `raw_area_px2`, `cleaned_area_px2` | Foreground pixel counts before/after cleanup |
| `cleaned_perimeter_px` | Length of the largest external contour in pixels; excludes hole boundaries |
| `cleaned_circularity` | `min(1, 4πA/P²)` using cleaned pixel area and external perimeter; missing below three foreground pixels |
| `tcmi_raw` | Mask-only Two-Circle Morphology Index, 0–100; higher indicates supported two-lobe morphology, not a probability. Empty/small masks return zero |
| `stage` | Predicted group, or `not_assessed_empty_mask`; present only with a staging model |
| `prob_<group>` | Uncalibrated classifier probability; missing for empty masks |

The four machine labels are `early_pre_implantation_hatching`,
`floating_apposition_adhesion`, `implantation_or_later`, and `degeneration`.
An empty mask does not establish biological absence.

## The 15 staging descriptors

These use the paper's definitions.
For shape, let `r=max(sqrt(A/π),1)` for the largest component. Appearance uses
RGB images downsampled to a maximum side of 384, per-channel 1st–99th percentile
normalization, and grayscale intensities in [0,1]. Boundary bands use a 7×7
erosion/dilation kernel. Undefined measurements remain missing for imputation
by the fitted classifier; missing boundary values are not filled during extraction.

| Column | Definition |
|---|---|
| `rich_deficit_component_count` | Number of connected regions in the filled convex hull minus the mask |
| `rich_skeleton_length_scaled` | Skeleton pixel count divided by `r` |
| `rich_radial_median` | Median external-contour distance from the foreground centroid, divided by `r` |
| `rich_core_fraction_075` | Fraction of foreground pixels with distance-to-background / `r` ≥ 0.75 |
| `rich_distance_max` | Maximum foreground distance-to-background / `r` |
| `rich_distance_std` | Population standard deviation of foreground distance-to-background / `r` |
| `texture_boundary_gray_contrast` | Mean grayscale intensity in the inner band minus the outer band |
| `texture_boundary_gray_absolute_contrast` | Absolute value of the preceding contrast |
| `texture_boundary_gradient_contrast` | Inner-minus-outer mean 3×3 Sobel gradient magnitude |
| `texture_gradient_std` | Population standard deviation of Sobel gradient magnitude within the mask |
| `texture_lbp_mean` | Mean eight-neighbor binary-pattern code, divided by 255 |
| `texture_lbp_uniform_fraction` | Fraction of codes in {0,255,1,2,4,8,16,32,64,128}; not the full conventional uniform-pattern set |
| `texture_glcm_homogeneity` | Sum of `p(i,j)/(1+(i−j)²)` |
| `texture_glcm_energy` | Sum of `p(i,j)²` (angular second moment, not its square root) |
| `texture_glcm_entropy` | `−sum(p log₂ p)` over positive co-occurrence probabilities |

Co-occurrence probabilities pool symmetric one-pixel pairs in four directions,
with 16 grayscale bins and both pixels inside the mask. Local binary patterns
compare neighbors against the center; the outermost image-border codes are zero.
Shape quantities are dimensionless; appearance quantities use normalized
intensity/code units, not physical units. Column order is fixed in `feature_set.json`.


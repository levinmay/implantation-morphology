"""Compare physical-area summaries before and after the 57-image exclusion."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROUGH = 'experiment_dpf_morphology_normalized_median'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def prepare(source):
    frame = pd.read_csv(source, keep_default_na=False)
    if not len(frame) == frame.unit_id.nunique() == 577:
        raise ValueError("Expected 577 unique images")
    if not (frame.embryo_id.nunique() == 131):
        raise ValueError("Calibration cohort differs from the paper")
    rough = frame.calibration_basis.eq(ROUGH)
    exp25 = frame.experiment_num.eq(25)
    if not (exp25.sum() == 68 and (exp25 & rough).sum() == 57):
        raise ValueError("Calibration cohort differs from the paper")
    if not ((exp25 & ~rough).sum() == 11):
        raise ValueError("Calibration cohort differs from the paper")
    if not (rough.sum() == 351):
        raise ValueError("Calibration cohort differs from the paper")
    for column in ['reference_area_um2', 'prediction_area_um2']:
        if not (np.isfinite(frame[column]).all() and frame[column].ge(0).all()):
            raise ValueError("Invalid physical area")
    selectors = {
        'all_existing_scales': pd.Series(True, index=frame.index),
        'exclude_exp25_57_rough_images': ~(exp25 & rough),
    }
    if not (selectors['exclude_exp25_57_rough_images'].sum() == 520):
        raise ValueError("Calibration cohort differs from the paper")
    return frame, selectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True,
                        help='Private 577-row calibration table; see MANIFEST_SCHEMA.md')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for aggregate summaries and private subset membership')
    args = parser.parse_args()
    source = args.input
    before = digest(source)
    frame, selectors = prepare(source)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / 'aggregate').mkdir()
    (output / 'private').mkdir()
    coverage, summaries, patterns = [], [], []
    membership = frame[['unit_id', 'embryo_id', 'experiment_num']].copy()
    for name, selector in selectors.items():
        subset = frame.loc[selector]
        membership[name] = selector
        coverage.append(dict(subset=name, images=len(subset), embryos=subset.embryo_id.nunique(),
                             physical_sources=subset.source_sha256.nunique(),
                             excluded_images=len(frame)-len(subset),
                             rough_images_retained=int(subset.calibration_basis.eq(ROUGH).sum())))
        for dimension in ['dpf', 'four_group']:
            for group_name, group in subset.groupby(dimension):
                for mask in ['reference', 'prediction']:
                    values = group[mask + '_area_um2']
                    summaries.append(dict(subset=name, dimension=dimension, group=group_name,
                        mask_source=mask, images=len(group), embryos=group.embryo_id.nunique(),
                        mean_area_um2=values.mean(), median_area_um2=values.median(),
                        sd_area_um2=values.std(ddof=1)))
        for mask in ['reference', 'prediction']:
            day = subset.groupby('dpf')[mask + '_area_um2'].mean()
            group = subset.groupby('four_group')[mask + '_area_um2'].mean()
            patterns.append(dict(subset=name, mask_source=mask,
                mean_area_increases_dpf8_9_10=bool(day.loc[8] < day.loc[9] < day.loc[10]),
                mean_area_decreases_dpf10_11=bool(day.loc[11] < day.loc[10]),
                largest_mean_area_group=group.idxmax(), smallest_mean_area_group=group.idxmin()))
    pd.DataFrame(coverage).to_csv(output / 'aggregate/COVERAGE.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'aggregate/AREA_SUMMARIES.csv', index=False)
    pd.DataFrame(patterns).to_csv(output / 'aggregate/PATTERNS.csv', index=False)
    membership.to_csv(output / 'private/SUBSET_MEMBERSHIP.csv', index=False)
    if before != digest(source):
        raise ValueError("Calibration input changed")
    result = dict(status='complete', input=str(source), input_sha256=before,
        script_sha256=digest(Path(__file__)), scales_changed=False, coverage=coverage,
        summary='Descriptive image-weighted means, matching the manuscript area summary; repeated embryos are not independent observations.',
        limitation='Existing calibration values are reused; this does not establish their accuracy.',
        patterns=patterns)
    (output / 'aggregate/RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    (output / 'SUCCESS').write_text('Validated calibration subset sensitivity complete\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()

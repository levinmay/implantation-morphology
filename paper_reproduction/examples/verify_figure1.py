"""Check characterization outputs against the approved Figure 1 example."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.io import read_mask

EXAMPLE = Path(__file__).with_name('figure1')


def verify(output: Path) -> None:
    expected = pd.read_csv(EXAMPLE / 'expected_measurements.csv').set_index('image').sort_index()
    actual = pd.read_csv(output / 'measurements.csv').set_index('image').sort_index()
    if not expected.index.equals(actual.index) or not expected.columns.equals(actual.columns):
        raise ValueError('Unexpected example rows or output fields')
    if not actual.status.equals(expected.status):
        raise ValueError('Example mask status differs')
    numeric = expected.select_dtypes(include='number').columns
    if not np.allclose(expected[numeric], actual[numeric], rtol=1e-7, atol=1e-8, equal_nan=True):
        raise ValueError('Example measurements differ from recorded values')
    for name in expected.index:
        if not np.array_equal(read_mask(EXAMPLE / 'cleaned_masks' / name),
                              read_mask(output / 'cleaned_masks' / name)):
            raise ValueError(f'Example cleanup differs: {name}')
    if not (output / 'CHECKSUMS.sha256').is_file():
        raise ValueError('Missing completed-output checksum ledger')
    print('PASS: six Figure 1 images, cleaned masks and measurements')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    verify(args.output)


if __name__ == '__main__':
    main()

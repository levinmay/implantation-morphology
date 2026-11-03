"""Check that the exclusion removes rough scales, not all experiment-25 images."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from paper_reproduction.experiments.calibration_sensitivity import prepare, ROUGH


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({
            'unit_id': [f'image_{i}' for i in range(577)],
            'embryo_id': [f'embryo_{i % 131}' for i in range(577)],
            'experiment_num': [25 if i < 68 else 1 for i in range(577)],
            'calibration_basis': [ROUGH if i < 57 or 68 <= i < 362 else 'recorded' for i in range(577)],
            'reference_area_um2': [100.] * 577, 'prediction_area_um2': [95.] * 577,
        })

    def load(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'calibration.csv'
            self.frame.to_csv(path, index=False)
            return prepare(path)

    def test_keep_the_eleven_recorded_scales_in_experiment_25(self):
        frame, selectors = self.load()
        kept = frame.loc[selectors['exclude_exp25_57_rough_images']]
        self.assertEqual(len(kept), 520)
        self.assertEqual(int(kept.experiment_num.eq(25).sum()), 11)
        self.assertEqual(kept.embryo_id.nunique(), 131)

    def test_reject_changed_exclusion_membership(self):
        self.frame.loc[0, 'calibration_basis'] = 'recorded'
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()

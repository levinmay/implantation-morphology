"""Check that outer-test information cannot affect training decisions."""
import unittest
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from paper_reproduction.experiments.feature_selection import fit_outer

CLASSES = ['a', 'b', 'c', 'd']


def candidates(features, outer):
    return [(str(c), Pipeline([('impute', SimpleImputer()),
             ('classify', LogisticRegression(C=c, max_iter=1000))])) for c in [0.1, 1.0]]


class IndependenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.frame = pd.DataFrame([
            {'unit_id': f'{fold}-{i}', 'component_id': f'{fold}-{i // 2}',
             'evaluation_fold': fold, 'target': i % 4, 'four_group': CLASSES[i % 4],
             'x0': i % 4 + rng.normal(), 'x1': rng.normal(), 'x2': rng.normal(),
             'x3': rng.normal(), 'constant': 2., 'missing': np.nan}
            for fold in range(5) for i in range(32)])
        self.columns = ['x0', 'x1', 'x2', 'x3', 'constant', 'missing']

    def run_fit(self, frame):
        return fit_outer(frame, self.columns, 2, 0, candidates, CLASSES,
                         trees=12, repeats=2, jobs=1)

    def test_outer_test_labels_do_not_change_any_fit_decision_or_prediction(self):
        first = self.run_fit(self.frame)
        changed = self.frame.copy()
        mask = changed.evaluation_fold == 0
        changed.loc[mask, 'target'] = (changed.loc[mask, 'target'] + 1) % 4
        second = self.run_fit(changed)
        self.assertEqual(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])
        np.testing.assert_array_equal(first[0].filter(like='prob_'), second[0].filter(like='prob_'))

    def test_outer_test_feature_values_cannot_change_selection_or_model(self):
        first = self.run_fit(self.frame)
        changed = self.frame.copy()
        changed.loc[changed.evaluation_fold == 0, ['x0', 'x1']] = 10000.
        second = self.run_fit(changed)
        self.assertEqual(first[1], second[1])
        pd.testing.assert_frame_equal(first[2], second[2])
        np.testing.assert_array_equal(first[5].named_steps['classify'].coef_,
                                      second[5].named_steps['classify'].coef_)

    def test_validation_is_excluded_from_ranking(self):
        first = self.run_fit(self.frame)
        changed = self.frame.copy()
        mask = changed.evaluation_fold == 1
        changed.loc[mask, 'target'] = (changed.loc[mask, 'target'] + 1) % 4
        second = self.run_fit(changed)
        pd.testing.assert_frame_equal(first[2], second[2])

    def test_cross_fold_group_is_rejected(self):
        changed = self.frame.copy()
        changed.loc[changed.evaluation_fold == 0, 'component_id'] = '1-0'
        with self.assertRaisesRegex(ValueError, 'crosses folds'):
            self.run_fit(changed)

    def test_constant_and_missing_training_features_are_excluded(self):
        result = self.run_fit(self.frame)
        self.assertEqual(result[1]['excluded_constant_or_missing'], ['constant', 'missing'])
        self.assertFalse(set(result[1]['features']) & {'constant', 'missing'})


if __name__ == '__main__':
    unittest.main()

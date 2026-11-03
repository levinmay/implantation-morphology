"""Check full candidate extraction against the supplied compact measurements."""
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from pipeline.cleanup import topology
from pipeline.features import default_feature_set, extract_features
from paper_reproduction.experiments.candidate_features import extract_candidates


class CandidateFeatureTests(unittest.TestCase):
    def test_full_pool_preserves_compact_features_on_all_example_images(self):
        examples = Path(__file__).resolve().parents[2] / 'examples/figure1'
        for path in sorted((examples / 'images').glob('*.png')):
            with self.subTest(image=path.name):
                image = np.asarray(Image.open(path).convert('RGB'))
                mask = topology(np.asarray(Image.open(examples / 'predicted_masks' / path.name)) > 0)
                whole, within, morphology = extract_candidates(image, mask)
                _, compact_appearance, compact_morphology = extract_features(image, mask, default_feature_set())
                self.assertEqual((len(whole), len(within), len(morphology)), (42, 42, 49))
                for name, value in {**compact_appearance, **compact_morphology}.items():
                    np.testing.assert_allclose({**within, **morphology}[name], value,
                                               rtol=1e-7, atol=1e-8, equal_nan=True)


if __name__ == '__main__':
    unittest.main()

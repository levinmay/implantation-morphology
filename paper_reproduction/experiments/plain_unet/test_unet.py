import unittest
import torch
from .unet import PlainUNet


class PlainUNetTests(unittest.TestCase):
    def test_native_shape_and_gradient_flow_through_all_levels(self):
        torch.set_num_threads(2)
        model = PlainUNet()
        image = torch.randn(1, 3, 63, 65)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (1, 1, 63, 65))
        logits.square().mean().backward()
        for part in (*model.encoders, *model.upconvs, *model.decoders, model.output):
            gradients = [p.grad for p in part.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.count_nonzero() > 0 for g in gradients))
        self.assertTrue(30_000_000 < sum(p.numel() for p in model.parameters()) < 32_000_000)


if __name__ == "__main__":
    unittest.main()

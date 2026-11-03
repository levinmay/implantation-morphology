import unittest
import numpy as np
from PIL import Image
import torch
from .model import preprocess, loss_function, native_prediction, augment


class SegmentationTests(unittest.TestCase):
    def test_aspect_ratio_padding_and_native_shape(self):
        image = Image.new("RGB", (120, 60), "white")
        mask = Image.new("L", (120, 60), 255)
        x, y, v, resized, native = preprocess(image, mask, 32)
        self.assertEqual(resized, (16, 32))
        self.assertEqual(tuple(x.shape), (3, 32, 32))
        self.assertEqual(v.sum().item(), 512)
        self.assertTrue(torch.equal(y, v))
        result = native_prediction(torch.ones(1, 1, 8, 8), resized, native, 32)
        self.assertEqual(result.shape, (60, 120))
        self.assertTrue(result.all())

    def test_padding_does_not_contribute_to_loss(self):
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        target = torch.ones_like(logits)
        valid = torch.ones_like(logits)
        valid[:, :, 4:] = 0
        original = loss_function(logits, target, valid)
        altered = logits.detach().clone()
        altered[:, :, 4:] = 100
        self.assertTrue(torch.equal(original, loss_function(altered, target, valid)))
        original.backward()
        self.assertEqual(logits.grad[:, :, 4:].count_nonzero().item(), 0)

    def test_augmentation_keeps_mask_and_image_aligned(self):
        image = torch.zeros(1, 16, 16)
        image[:, 2:5, 7:12] = 1
        for seed in range(20):
            x, y, _ = augment(image, image.clone(), torch.ones_like(image), np.random.default_rng(seed))
            self.assertTrue(torch.equal(x, y))
            self.assertEqual(x.sum(), image.sum())


    def test_unsupported_gpu_stops_before_loading_inputs_or_creating_outputs(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from . import run

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new-run"
            arguments = ["plain-unet", "--project-root", "missing", "--release-dir", "missing",
                         "--split-dir", "missing", "--nnunet-dir", "missing",
                         "--output-dir", str(output)]
            for cuda, bf16 in ((False, False), (True, False)):
                with self.subTest(cuda=cuda, bf16=bf16):
                    message = io.StringIO()
                    with patch("sys.argv", arguments), patch.object(run, "prepare") as prepare, \
                            patch.object(run.torch.cuda, "is_available", return_value=cuda), \
                            patch.object(run.torch.cuda, "is_bf16_supported", return_value=bf16), \
                            contextlib.redirect_stderr(message):
                        with self.assertRaises(SystemExit) as error:
                            run.main()
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn("requires a CUDA GPU with bfloat16 support", message.getvalue())
                    prepare.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

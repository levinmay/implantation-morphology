"""Aspect-preserving preprocessing and binary segmentation loss/inference."""
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

SIZE = 512
MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def preprocess(image, mask, size=SIZE):
    if image.size != mask.size:
        raise ValueError("Image and reference grids differ")
    width, height = image.size
    scale = size / max(height, width)
    rh, rw = max(1, round(height * scale)), max(1, round(width * scale))
    rgb = np.asarray(image.convert("RGB").resize((rw, rh), Image.Resampling.BILINEAR)).copy()
    target = np.asarray(mask.convert("L").resize((rw, rh), Image.Resampling.NEAREST)).copy() > 0
    image_tensor = (torch.from_numpy(rgb).permute(2, 0, 1).float() / 255 - MEAN) / STD
    target_tensor = torch.from_numpy(target).float()[None]
    valid = torch.ones_like(target_tensor)
    padding = (0, size - rw, 0, size - rh)
    return (F.pad(image_tensor, padding), F.pad(target_tensor, padding),
            F.pad(valid, padding), (rh, rw), (height, width))


def loss_function(logits, target, valid):
    logits = F.interpolate(logits.float(), target.shape[-2:], mode="bilinear", align_corners=False)
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * valid).sum((1, 2, 3)) / valid.sum((1, 2, 3)).clamp_min(1)
    probability = logits.sigmoid() * valid
    target = target * valid
    dice = 1 - (2 * (probability * target).sum((1, 2, 3)) + 1) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return (bce + dice).mean()


def native_prediction(logits, resized_shape, native_shape, size=SIZE):
    grid = F.interpolate(logits.float(), (size, size), mode="bilinear", align_corners=False)
    rh, rw = resized_shape
    grid = grid[:, :, :rh, :rw]
    grid = F.interpolate(grid, native_shape, mode="bilinear", align_corners=False)
    return grid[0, 0].detach().cpu().numpy() > 0


def augment(image, target, valid, generator):
    rotations = int(generator.integers(4))
    tensors = [torch.rot90(x, rotations, (-2, -1)) for x in (image, target, valid)]
    if generator.random() < 0.5:
        tensors = [x.flip(-1) for x in tensors]
    if generator.random() < 0.5:
        tensors = [x.flip(-2) for x in tensors]
    return tensors


"""Seed Python, NumPy, and PyTorch before initializing the standard trainer."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_seeded(nnUNetTrainer):
    """Use NNUNET_SEED (default 42); inherit all training settings."""

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        seed = int(os.environ.get("NNUNET_SEED", "42"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        super().__init__(plans, configuration, fold, dataset_json, device)
        self.seed = seed

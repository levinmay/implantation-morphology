"""Run the seeded nnU-Net trainer for 250 epochs."""

from __future__ import annotations

import torch

from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_seeded import (
    nnUNetTrainer_seeded,
)


class nnUNetTrainer_250epochs_seeded(nnUNetTrainer_seeded):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250

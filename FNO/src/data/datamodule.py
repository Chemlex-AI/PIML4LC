from __future__ import annotations

from omegaconf import DictConfig
from typing import Literal

from torch.utils.data import DataLoader

import lightning.pytorch as pl

__all__ = [
    'DataModule'
]

class DataModule(pl.LightningDataModule):
    def __init__(
            self,
            loader_cfg: DictConfig,
            collator = None
        ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.loader_cfg = loader_cfg
        self.collator = collator

    def setup(self, stage: Literal['fit', 'validate', 'test', 'predict']):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            collate_fn=self.collator,
            **self.loader_cfg
        )

    def val_dataloader(self):
        cfg = {**self.loader_cfg}
        cfg['shuffle'] = False
        return DataLoader(
            dataset=self.valid_dataset,
            collate_fn=self.collator,
            **cfg
        )

    def test_dataloader(self):
        cfg = {**self.loader_cfg}
        cfg['shuffle'] = False
        return DataLoader(
            dataset=self.test_dataset,
            collate_fn=self.collator,
            **cfg
        )



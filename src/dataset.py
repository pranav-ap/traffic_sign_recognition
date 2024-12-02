import os
from typing import Dict

import lightning as L
import numpy as np
import torch
import torchvision
from torchvision import transforms as T
from torch.utils.data import Subset, DataLoader
from config import config
from utils import logger

torch.set_float32_matmul_precision('medium')


class TrafficSignsDataModule(L.LightningDataModule):
    def __init__(self):
        super().__init__()

        self.num_workers = 0 if config.task.eda_mode else os.cpu_count()
        self.persistent_workers = not config.task.eda_mode

        self.transform = T.Compose([
            T.Resize((config.image.image_size, config.image.image_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

        self.datasets: Dict[str, Subset] = {}

    def setup(self, stage=None):
        if stage == "fit":
            train_data = torchvision.datasets.GTSRB(
                root=config.paths.roots.data,
                split='train',
                download=True,
                transform=self.transform,
            )

            labels = [i for i in range(43)]

            from sklearn.model_selection import train_test_split
            train_indices, val_indices = train_test_split(
                np.arange(len(labels)),
                test_size=0.2,
                # stratify=labels,
            )

            self.datasets['train'] = Subset(train_data, train_indices)
            self.datasets['val'] = Subset(train_data, val_indices)

            logger.info(f"Train Dataset       : {len(self.datasets['train'])} samples")
            logger.info(f"Validation Dataset  : {len(self.datasets['val'])} samples")

    def train_dataloader(self):
        return DataLoader(
            self.datasets['train'],
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.datasets['val'],
            batch_size=config.train.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=True,
        )

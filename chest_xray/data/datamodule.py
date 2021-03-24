import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .dataset import InferenceDicomDataset, XRayAnomalyDataset, XRayAnomalyValidationDataset
from .transforms import test_transform, train_transform
from ..conf.data import DataConfig

__all__ = [
    'XRayDataModule',
]


class XRayDataModule(LightningDataModule):

    def __init__(self, data_config: DataConfig) -> None:
        super().__init__(
            train_transforms=train_transform(data_config.train.crop.width, data_config.train.crop.height),
            val_transforms=test_transform(),
            test_transforms=test_transform()
        )
        self.config = data_config
        self.train_data_dir = Path(data_config.train.data_dir)
        self.train_metadata = Path(data_config.train.metadata)
        self.validation_data_dir = Path(data_config.validation.data_dir)
        self.validation_metadata = Path(data_config.validation.metadata)
        self.predict_data_dir = Path(data_config.predict.data_dir)
        self.train_dataset: XRayAnomalyDataset = None
        self.validation_dataset: XRayAnomalyValidationDataset = None
        self.predict_dataset: InferenceDicomDataset = None

    def prepare_data(self, *args, **kwargs) -> None:
        os.system(f'dvc pull {os.fspath(self.train_data_dir)}')
        os.system(f'dvc pull {os.fspath(self.predict_data_dir)}')

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = XRayAnomalyDataset(self.train_data_dir, self.train_metadata, self.train_transforms)
        self.validation_dataset = XRayAnomalyValidationDataset(
            self.validation_data_dir,
            self.validation_metadata,
            self.val_transforms,
            ignore_images_without_objects=False,  # TODO: Check me
        )
        self.predict_dataset = InferenceDicomDataset(self.predict_data_dir, self.test_transforms)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.loader.batch_size,
            shuffle=True,
            num_workers=self.config.train.loader.num_workers,
            pin_memory=True,
            collate_fn=train_collate_fn
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.config.validation.loader.batch_size,
            num_workers=self.config.validation.loader.num_workers,
            pin_memory=True,
            collate_fn=validation_collate_fn
        )

    def test_dataloader(self) -> DataLoader:
        # It's a work around the fact that currently I can't compute mAP in DDP mode
        return self.val_dataloader()

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.config.predict.loader.batch_size,
            num_workers=self.config.predict.loader.num_workers,
            pin_memory=True,
            collate_fn=test_collate_fn
        )


def train_collate_fn(
    batch: List[Dict[str, Union[np.array, float]]],
    image_field: str = 'image',
    boxes_field: str = 'boxes'
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    images = []
    targets = []
    for example in batch:
        image = example.pop(image_field)
        images.append(image)
        target = {}
        for key, value in example.items():
            value = torch.from_numpy(np.array(value))
            if key == boxes_field:
                value = value.float()
            target[key] = value
        targets.append(target)
    return images, targets


def validation_collate_fn(
    batch: List[Dict[str, Union[np.array, float]]],
    image_field: str = 'image',
    boxes_field: str = 'boxes'
) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    indices = []
    examples = []
    for index,  example in batch:
        indices.append(index)
        examples.append(example)
    indices = torch.tensor(indices)
    images, targets = train_collate_fn(examples, image_field, boxes_field)
    return indices, images, targets


def test_collate_fn(batch: List[Tuple[int, torch.Tensor]]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    indices = []
    images = []
    for index, image in batch:
        indices.append(index)
        images.append(image)
    indices = torch.tensor(indices)
    return indices, images

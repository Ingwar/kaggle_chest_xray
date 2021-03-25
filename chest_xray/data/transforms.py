from typing import Any, List

import numpy as np
from albumentations import BasicTransform, BboxParams, Compose, Flip, GaussNoise, Lambda, RandomBrightnessContrast, \
    RandomRotate90, \
    RandomSizedBBoxSafeCrop, ToFloat
from albumentations.pytorch import ToTensorV2

__all__ = [
    'train_transform',
    'train_aggressive_transform',
    'test_transform',
]


def train_transform(
    from_dicom: bool,
    crop_width: int,
    crop_height: int,
    additional_transforms: List[BasicTransform] = None
) -> Compose:
    additional_transforms = additional_transforms if additional_transforms is not None else []
    initial_transforms = [
        RandomSizedBBoxSafeCrop(width=crop_width, height=crop_height, erosion_rate=0.2),
        # LongestMaxSize(800),
    ]
    if from_dicom:
        initial_transforms.insert(0, Lambda(image=stack_channels_for_rgb))
    final_transforms = [
        ToFloat(),
        ToTensorV2(),
    ]
    transforms = initial_transforms + additional_transforms + final_transforms
    return Compose(transforms, bbox_params=BboxParams(format='pascal_voc', label_fields=['labels']))


def train_aggressive_transform(from_dicom: bool, crop_width: int, crop_height: int) -> Compose:
    additional_transforms = [
        Flip(),
        RandomRotate90(),
        # ShiftScaleRotate(),
        RandomBrightnessContrast(),
        GaussNoise(),
    ]
    return train_transform(from_dicom, crop_width, crop_height, additional_transforms)


def test_transform(from_dicom: bool) -> Compose:
    transforms = [
        ToFloat(),
        ToTensorV2(),
    ]
    if from_dicom:
        transforms.insert(0, Lambda(image=stack_channels_for_rgb))
    return Compose(transforms)


def stack_channels_for_rgb(image: np.ndarray, **kwargs: Any) -> np.ndarray:
    if len(image.shape) != 2:
        raise ValueError(f'Expected single-channel image, but got {image.shape}')
    return np.stack([image, image, image], axis=-1)

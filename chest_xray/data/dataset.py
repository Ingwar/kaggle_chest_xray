import os
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import AnyStr, BinaryIO, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydicom
import torch
from albumentations import BasicTransform, Compose
from pydicom.pixel_data_handlers import apply_voi_lut
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader

__all__ = [
    'XRayAnomalyDataset',
    'XRayAnomalyValidationDataset',
    'InferenceDataset',
    'InferenceDicomDataset',
    'InferencePngDataset',
]


class XRayAnomalyDataset(Dataset):

    def __init__(
        self,
        data_dir: Path,
        metadata_file: Path,
        transform: Optional[Union[Compose, BasicTransform]] = None,
        ignore_images_without_objects: bool = True,
        background_class: int = 14,
        read_dicom: bool = True,
    ) -> None:
        metadata = pd.read_csv(metadata_file)
        if ignore_images_without_objects:
            metadata = metadata[metadata['class_id'] != background_class]
        if read_dicom:
            suffix = 'dicom'
        else:
            suffix = 'png'
        self.image_ids = pd.unique(metadata['image_id'])
        self.file_list = [data_dir / f'{image_id}.{suffix}' for image_id in self.image_ids]
        self.metadata = metadata
        self.transform = transform
        self.background_class = background_class
        self.read_dicom = read_dicom

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_file = self.file_list[index]
        image_id = image_file.stem
        image_metadata = self.metadata[self.metadata['image_id'] == image_id]
        boxes = []
        labels = []
        for row in image_metadata[image_metadata['class_id'] != self.background_class].itertuples(index=False):
            labels.append(row.class_id + 1)  # Torchvision expects that 0 is a background class
            boxes.append([row.x_min, row.y_min, row.x_max, row.y_max])
        labels = np.array(labels)
        boxes = np.array(boxes)
        if self.read_dicom:
            image = read_xray(image_file)
        else:
            image = np.array(pil_loader(image_file))
        result = {'image': image, 'boxes': boxes, 'labels': labels}
        if self.transform is not None:
            result = self.transform(image=image, bboxes=boxes, labels=labels)
            # Rename bounding boxes field to match torchvision expectations
            boxes = result.pop('bboxes')
            result['boxes'] = boxes
        return result

    def __len__(self) -> int:
        return len(self.file_list)


class XRayAnomalyValidationDataset(XRayAnomalyDataset):

    def __getitem__(self, index: int) -> Tuple[int, Dict[str, torch.Tensor]]:
        return index, super().__getitem__(index)


class InferenceDataset(Dataset, metaclass=ABCMeta):

    def __init__(self, image_dir: Path, transform: Optional[Union[Compose, BasicTransform]] = None) -> None:
        self.transform = transform
        self.file_list = list(image_dir.glob('*'))
        self.image_ids = [image_file.stem for image_file in self.file_list]

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        image_file = self.file_list[index]
        image = self._read_image(image_file)
        if self.transform is not None:
            image = self.transform(image=image)['image']
        return index, image

    def __len__(self) -> int:
        return len(self.file_list)

    @abstractmethod
    def _read_image(self, image_file: Path) -> np.ndarray:
        pass


class InferenceDicomDataset(InferenceDataset):

    def _read_image(self, image_file: Path) -> np.ndarray:
        return read_xray(image_file)


class InferencePngDataset(InferenceDataset):

    def _read_image(self, image_file: Path) -> np.ndarray:
        return np.array(pil_loader(image_file))


# Code by raddar from https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
def read_xray(path: Union[str, 'os.PathLike[AnyStr]', BinaryIO], voi_lut: bool = True, fix_monochrome: bool = True) -> np.ndarray:
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data


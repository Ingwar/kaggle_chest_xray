from dataclasses import dataclass
from typing import Optional

from omegaconf import MISSING

__all__ = [
    'DataConfig',
]


@dataclass
class DataLoadingConfig:
    batch_size: int = MISSING
    num_workers: int = MISSING
    load_dicom: bool = MISSING


@dataclass
class TrainDataConfig:
    data_dir: str = MISSING
    metadata: str = MISSING
    longest_size: int = MISSING
    ignore_images_without_objects: bool = True
    augment: bool = MISSING
    use_mixup: bool = False
    mixup_warmup_epochs: int = 0
    loader: DataLoadingConfig = DataLoadingConfig()


@dataclass
class ValidationDataConfig:
    data_dir: str = MISSING
    metadata: str = MISSING
    loader: DataLoadingConfig = DataLoadingConfig()


@dataclass
class PredictDataConfig:
    data_dir: str = MISSING
    loader: DataLoadingConfig = DataLoadingConfig()


@dataclass
class DataConfig:
    train: TrainDataConfig = TrainDataConfig()
    validation: Optional[ValidationDataConfig] = ValidationDataConfig()
    predict: PredictDataConfig = PredictDataConfig()
    default_loader: DataLoadingConfig = DataLoadingConfig()


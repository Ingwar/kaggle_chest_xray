from dataclasses import dataclass

from omegaconf import MISSING

__all__ = [
    'DataConfig',
]


@dataclass
class CropInfo:
    width: int = MISSING
    height: int = MISSING


@dataclass
class DataLoadingConfig:
    batch_size: int = MISSING
    num_workers: int = MISSING


@dataclass
class TrainDataConfig:
    data_dir: str = MISSING
    metadata: str = MISSING
    crop: CropInfo = CropInfo()
    loader: DataLoadingConfig = DataLoadingConfig()


@dataclass
class ValidationDataConfig:
    data_dir: str = MISSING
    metadata: str = MISSING
    loader: DataLoadingConfig = DataLoadingConfig()


@dataclass
class TestDataConfig:
    data_dir: str = MISSING
    loader: DataLoadingConfig = DataLoadingConfig()


@dataclass
class DataConfig:
    train: TrainDataConfig = TrainDataConfig()
    validation: ValidationDataConfig = ValidationDataConfig()
    test: TestDataConfig = TestDataConfig()
    default_loader = DataLoadingConfig()


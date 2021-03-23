from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import MISSING

from .data import DataConfig

__all__ = [
    'PipelineConfig',
]


@dataclass
class ModelConfig:
    num_classes: int = MISSING
    trainable_backbone_layers: int = MISSING


@dataclass
class PipelineConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimizer: Dict[str, Any] = MISSING


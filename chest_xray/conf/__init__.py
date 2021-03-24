from dataclasses import dataclass
from typing import Any, Dict

from omegaconf import MISSING

from .data import DataConfig

__all__ = [
    'PipelineConfig',
]


@dataclass
class ModelConfig:
    backbone: str = MISSING
    num_classes: int = MISSING
    trainable_backbone_layers: int = MISSING


@dataclass
class SubmissionConfig:
    file: str = MISSING


@dataclass
class EvaluationConfig:
    iou_threshold: float = MISSING
    confidence_threshold: float = MISSING


@dataclass
class PipelineConfig:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimizer: Dict[str, Any] = MISSING
    submission: SubmissionConfig = SubmissionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


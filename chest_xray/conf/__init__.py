from dataclasses import dataclass, field
from typing import Any, Dict, List

from omegaconf import MISSING

from .data import DataConfig

__all__ = [
    'PipelineConfig',
]

from .scheduler import SchedulerConf


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
    schedulers: List[SchedulerConf] = field(default_factory=list)
    submission: SubmissionConfig = SubmissionConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

from dataclasses import  dataclass
from typing import Any, Dict, Optional

from omegaconf import MISSING

__all__ = [
    'SchedulerConf',
]


@dataclass
class SchedulerConf:
    scheduler: Dict[str, Any] = MISSING
    interval: str = 'epoch'
    frequency: int = 1
    reduce_on_plateau: bool = False
    monitor: str = 'val_loss'
    strict: bool = True
    name: Optional[str] = None

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim import Optimizer

from chest_xray.conf import PipelineConfig

__all__ = [
    'Experiment',
]

TrainBatch = Tuple[List[torch.Tensor],  List[Dict[str, torch.Tensor]]]
TestBatch = List[torch.Tensor]
InferenceBatch = Tuple[torch.tensor, List[torch.Tensor]]

BatchLosses = Dict[str, torch.Tensor]
BatchPredictions = List[Dict[str, torch.Tensor]]


class Experiment(LightningModule):

    def __init__(self, model: nn.Module, conf: PipelineConfig) -> None:
        super().__init__()
        self.model = model
        self.conf = conf
        self.save_hyperparameters(conf)

    def forward(self, batch: Union[TrainBatch, TestBatch]) -> Union[BatchLosses, BatchPredictions]:
        if isinstance(batch, tuple):
            images, targets = batch
        else:
            images = batch
            targets = None
        return self.model(images, targets)

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        loss_dict = self(batch)
        loss = sum(loss_dict.values())
        return loss

    def validation_step(self, batch: TrainBatch, batch_idx: int) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        images, targets = batch
        predictions = self(images)
        return predictions, targets

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        # TODO: compute mAP
        self.log('mAP', -1, prog_bar=True)

    def predict(self, batch: InferenceBatch, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[torch.Tensor, BatchPredictions]:
        indices, images = batch
        predictions = self(images)
        return indices, predictions

    def configure_optimizers(self) -> Optimizer:
        return instantiate(self.conf.optimizer, self.parameters())

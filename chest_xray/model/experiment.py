from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from hydra.utils import instantiate
from map_boxes import mean_average_precision_for_boxes
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torch.optim import Optimizer

from . import instantiate_model
from ..conf import PipelineConfig

__all__ = [
    'Experiment',
]
WARMUP_SCHEDULER = 'chest_xray.model.schedulers.CosineDecayWithWarmupScheduler'

MetadataDict = Dict[str, torch.Tensor]

TrainBatch = Tuple[List[torch.Tensor],  List[MetadataDict]]
ValidationBatch = Tuple[torch.Tensor, List[torch.Tensor],  List[MetadataDict]]
TestBatch = List[torch.Tensor]
InferenceBatch = Tuple[torch.tensor, List[torch.Tensor]]

BatchLosses = Dict[str, torch.Tensor]
BatchPredictions = List[MetadataDict]

EMPTY_BOX = [0, 0, 1, 1]


class Experiment(LightningModule):

    def __init__(self, conf: PipelineConfig) -> None:
        super().__init__()
        self.model = instantiate_model(
            conf.model.backbone,
            conf.model.num_classes,
            conf.model.trainable_backbone_layers
        )
        self.conf = conf
        self.save_hyperparameters(conf)

    @property
    def no_findings_label(self) -> int:
        return self.conf.model.num_classes - 1

    def forward(self, batch: Union[TrainBatch, TestBatch]) -> Union[BatchLosses, BatchPredictions]:
        if isinstance(batch, tuple):
            images, targets = batch
        else:
            images = batch
            targets = None
        return self.model(images, targets)

    def training_step(self, batch: TrainBatch, batch_idx: int) -> torch.Tensor:
        loss_dict = self(batch)
        for loss_component_name, loss_component_value in loss_dict.items():
            self.log(loss_component_name, loss_component_value)
        loss = sum(loss_dict.values())
        self.log('Train loss', loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch: ValidationBatch, batch_idx: int) -> Tuple[torch.Tensor, List[MetadataDict], List[MetadataDict]]:
        indices, images, targets = batch
        predictions = self(images)
        return indices, predictions, targets

    def test_step(self, batch: ValidationBatch, batch_idx: int) -> Tuple[torch.Tensor, List[MetadataDict], List[MetadataDict]]:
        return self.validation_step(batch, batch_idx)

    def validation_epoch_end(self, outputs: List[Tuple[List[MetadataDict], List[MetadataDict]]]) -> None:
        predictions, targets = self._restructure_validation_outputs(outputs)
        mAP, ap_per_class = mean_average_precision_for_boxes(
            targets,
            predictions,
            iou_threshold=self.conf.evaluation.iou_threshold,
            verbose=False
        )
        # TODO: Add DDP support
        self.log('mAP', mAP, prog_bar=True)
        self.log_ap_per_class(ap_per_class)

    def test_epoch_end(self, outputs: List[Tuple[List[MetadataDict], List[MetadataDict]]]) -> None:
        return self.validation_epoch_end(outputs)

    def predict(self, batch: InferenceBatch, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[torch.Tensor, BatchPredictions]:
        indices, images = batch
        predictions = self(images)
        return indices, predictions

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = instantiate(self.conf.optimizer, self.parameters())
        schedulers = []
        for scheduler_conf in self.conf.schedulers:
            scheduler_conf = OmegaConf.to_container(scheduler_conf, resolve=True)
            if scheduler_conf['scheduler']['_target_'] == WARMUP_SCHEDULER:
                scheduler = instantiate(scheduler_conf['scheduler'], optimizer, total_epochs=self.trainer.max_epochs)
            else:
                scheduler = instantiate(scheduler_conf['scheduler'], optimizer)
            scheduler_conf['scheduler'] = scheduler
            schedulers.append(scheduler_conf)
        return [optimizer], schedulers

    def log_ap_per_class(self, ap_per_class: Dict[str, Tuple[float, float]]) -> None:
        for label, (average_precision, num_annotations) in ap_per_class.items():
            self.log(f'AP for class {label}', average_precision)
            self.log(f'Number of annotations for class {label}', num_annotations)

    def _restructure_validation_outputs(self, outputs: List[Tuple[List[MetadataDict], List[MetadataDict]]]) -> Tuple[np.ndarray, np.ndarray]:
        overall_predictions = []
        overall_targets = []
        for batch_indices, batch_predictions, batch_targets in outputs:
            assert len(batch_indices) == len(batch_predictions) == len(batch_targets)
            for index, prediction, target in zip(batch_indices.cpu().numpy(), batch_predictions, batch_targets):
                image_id = self.trainer.datamodule.validation_dataset.image_ids[index]
                predicted_boxes = prediction['boxes'].cpu().numpy()
                predicted_labels = prediction['labels'].cpu().numpy() - 1  # Go back to the original label range
                predicted_scores = prediction['scores'].cpu().numpy()
                if np.all(predicted_scores < self.conf.evaluation.confidence_threshold):
                    overall_predictions.append([image_id, self.no_findings_label, 1, *EMPTY_BOX])
                else:
                    for box, label, score in zip(predicted_boxes, predicted_labels, predicted_scores):
                        if score > self.conf.evaluation.confidence_threshold:
                            overall_predictions.append([image_id, label, score, *box])
                target_boxes = target['boxes'].cpu().numpy()
                target_labels = target['labels'].cpu().numpy()
                if len(target_boxes) == len(target_labels) == 0:
                    overall_targets.append([image_id, self.no_findings_label, *EMPTY_BOX])
                else:
                    for box, label in zip(target_boxes, target_labels):
                        overall_targets.append([image_id, label, *box])
        return np.array(overall_predictions), np.array(overall_targets)

from copy import copy
from typing import Dict, List, Tuple

import numpy as np
import torch
from pytorch_lightning import Callback, LightningModule, Trainer

__all__ = [
    'MixupData',
]

ImageWithMetadata = Tuple[torch.Tensor, Dict[str, torch.Tensor]]


# Inspired by the solution from  https://www.kaggle.com/c/global-wheat-detection/discussion/172418
class MixupData(Callback):

    def __init__(self, warmup_epochs: int) -> None:
        self.warmup_epochs = warmup_epochs

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Tuple[List[torch.Tensor],  List[Dict[str, torch.Tensor]]],
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        if trainer.current_epoch >= self.warmup_epochs and np.random.random() > 0.5:
            images, metadata = batch
            num_examples = len(images)
            image_copy = copy(images)
            metadata_copy = copy(metadata)
            shuffled_indices = np.random.permutation(num_examples)
            lambda_coef = np.random.beta(1.5, 1.5)
            for i in range(num_examples):
                image2_index = shuffled_indices[i]
                if i == image2_index:
                    continue
                image1 = image_copy[i]
                metadata1 = metadata_copy[i]
                image2 = image_copy[image2_index]
                metadata2 = metadata_copy[image2_index]
                mixed_image, mixed_metadata = mixup_two_images((image1, metadata1), (image2, metadata2), lambda_coef)
                images[i] = mixed_image
                metadata[i] = mixed_metadata


def mixup_two_images(image1: ImageWithMetadata, image2: ImageWithMetadata, lambda_coeff: float) -> ImageWithMetadata:
    image1, metadata1 = image1
    image2, metadata2 = image2

    assert set(metadata1.keys()) == {'labels', 'boxes'}
    assert set(metadata2.keys()) == {'labels', 'boxes'}

    new_height = max(image1.shape[1], image2.shape[1])
    new_width = max(image1.shape[2], image2.shape[2])

    mixed_image = torch.zeros(3, new_height, new_width)
    mixed_image[:, :image1.shape[1], :image2.shape[2]] = image1 * lambda_coeff
    mixed_image[:, :image1.shape[2], :image2.shape[2]] += image2 * (1 - lambda_coeff)
    mixed_labels = torch.cat([metadata1['labels'], metadata2['labels']], dim=0)
    mixed_boxes = torch.cat([metadata1['boxes'], metadata2['boxes']], dim=0)
    mixed_metadata = {'label': mixed_labels, 'boxes': mixed_boxes}
    return mixed_image, mixed_metadata

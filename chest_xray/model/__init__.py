import torchvision
from torch import nn

__all__ = [
    'instantiate_model',
]


def instantiate_model(num_classes: int) -> nn.Module:
    # TODO: make configurable
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)

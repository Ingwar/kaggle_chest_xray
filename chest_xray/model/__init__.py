from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

__all__ = [
    'instantiate_model',
]


def instantiate_model(backbone_name: str, num_classes: int, trainable_backbone_layers: int) -> FasterRCNN:
    # TODO: make configurable
    backbone = resnet_fpn_backbone(
        backbone_name,
        pretrained=True,
        trainable_layers=trainable_backbone_layers,
        # norm_layer=nn.BatchNorm2d,
    )
    return FasterRCNN(backbone, num_classes)

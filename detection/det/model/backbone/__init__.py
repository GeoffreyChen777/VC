from model.backbone.resnet import *
from model.block.block import FrozenBatchNorm2d

backbones = {"resnet50": resnet50, "resnet101": resnet101}


def make_backbone(cfg):
    backbone = backbones[cfg.model.backbone.name](
        pretrained=cfg.model.backbone.pretrained, norm_layer=FrozenBatchNorm2d
    )

    trainable_layers = cfg.model.backbone.trainable_layers
    assert (
        5 >= trainable_layers >= 0 and cfg.model.backbone.pretrained
    ), "Trainable layer can only set from 0 to 5 with pretrained = True"
    if cfg.model.backbone.pretrained:
        layers_to_train = ["res5", "res4", "res3", "res2", "stem"][:trainable_layers]
        for name, parameter in backbone.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

    return backbone

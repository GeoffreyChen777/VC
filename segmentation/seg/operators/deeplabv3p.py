import torch
from torch.nn.parallel import DistributedDataParallel

from gvcore.operators import OPERATOR_REGISTRY, GenericOpt
from gvcore.evaluator import EVALUATOR_REGISTRY
import gvcore.utils.distributed as dist_utils

from model.segmentor.deeplabv3p import DeeplabV3p


@OPERATOR_REGISTRY.register("deeplabv3p")
class DeeplabV3pOpt(GenericOpt):
    def __init__(self, cfg):
        super(DeeplabV3pOpt, self).__init__(cfg)

    def build_model(self):
        model = DeeplabV3p(self.cfg)
        model.to(self.device)
        if dist_utils.is_distributed():
            model = DistributedDataParallel(model)
        self.model = model

    def build_evaluator(self):
        if self.evaluator is None:
            self.evaluator = EVALUATOR_REGISTRY[self.cfg.evaluator](
                self.cfg.model.num_classes, distributed=dist_utils.is_distributed()
            )
        self.evaluator.reset()

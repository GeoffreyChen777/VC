import torch
from torch.nn.parallel import DistributedDataParallel

import gvcore.utils.distributed as dist_utils
from model.detector.virtual_category import VCDetector
from model.detector.virtual_category_cross import VCCrossTSDetector
from operators.virtual_category import VirtualCategoryOpt

from gvcore.operators import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register("virtual_category_cross")
class VirtualCategoryCrossOpt(VirtualCategoryOpt):
    def __init__(self, cfg):
        super(VirtualCategoryCrossOpt, self).__init__(cfg)

    def build_model(self):
        model = VCCrossTSDetector(self.cfg, VCDetector)
        model.to(self.device)
        if dist_utils.is_distributed():
            model = DistributedDataParallel(model)
        self.model = model

    def save_ckp(self, name=None):
        self.checkpointer.save(
            name=name if name is not None else self.cur_step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            step=self.cur_step,
        )

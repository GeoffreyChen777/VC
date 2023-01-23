import torch
from torch.nn.parallel import DistributedDataParallel

import gvcore.utils.distributed as dist_utils
from model.detector.virtual_category import VCDetector, VCTSDetector
from operators.unbiased_teacher import UnbiasedTeacherOpt

from gvcore.operators import OPERATOR_REGISTRY


@OPERATOR_REGISTRY.register("virtual_category")
class VirtualCategoryOpt(UnbiasedTeacherOpt):
    def __init__(self, cfg):
        super(VirtualCategoryOpt, self).__init__(cfg)

    def build_model(self):
        model = VCTSDetector(self.cfg, VCDetector)
        model.to(self.device)

        if dist_utils.is_distributed():
            model = DistributedDataParallel(model)
        self.model = model

    def resume_ckp(self, test_mode=False):
        super(VirtualCategoryOpt, self).resume_ckp(test_mode=test_mode)
        if self.cfg.resume is None:
            return
        if not test_mode:
            ckp = torch.load(self.cfg.resume, map_location=self.device)
            if "history" in ckp:
                history = ckp["history"]
                if dist_utils.is_distributed():
                    self.model.module.history.dict = history
                else:
                    self.model.history.dict = history

    def save_ckp(self, name=None):
        self.checkpointer.save(
            name=name if name is not None else self.cur_step,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            step=self.cur_step,
            history=self.model.module.history.dict if dist_utils.is_distributed() else self.model.history.dict,
        )

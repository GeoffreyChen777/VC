import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel
from addict import Dict

from gvcore.optim.utils import sum_loss
import gvcore.utils.distributed as dist_utils
from gvcore.operators import OPERATOR_REGISTRY

from gvcore.optim import make_lr_scheduler, make_optimizer
from model.segmentor.deeplabv3p_alt import DeeplabV3pAlt
from model.segmentor.vc import VCSegmentor
from operators.ts import TeacherStudentOpt


@OPERATOR_REGISTRY.register("virtual_category")
class VirtualCategoryOpt(TeacherStudentOpt):
    def __init__(self, cfg):
        super(VirtualCategoryOpt, self).__init__(cfg)

        self.generator_optimizer = None

    def build_model(self):
        model = VCSegmentor(self.cfg, DeeplabV3pAlt)
        model.to(self.device)
        if dist_utils.is_distributed():
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model)

        self.model = model

    def build_optimizer(self):
        self.optimizer = make_optimizer(
            self.cfg.solver.optimizer,
            self.model.module.segmentor if dist_utils.is_distributed() else self.model.segmentor,
        )
        self.generator_optimizer = make_optimizer(
            self.cfg.solver.optimizer,
            self.model.module.weight_generator if dist_utils.is_distributed() else self.model.weight_generator,
        )

        self.lr_scheduler = make_lr_scheduler(self.cfg.solver.lr_scheduler, self.optimizer)

    def train_pre_step(self) -> Dict:
        self.generator_optimizer.zero_grad()
        return super().train_pre_step()

    def train_run_step(self, model_inputs: Dict, **kwargs) -> Dict:
        loss_dict, stat_dict = self.model(**model_inputs)

        losses = sum_loss(loss_dict.loss_l) + (
            self.cfg.solver.u_loss_weight if self.cur_step > self.burnin_step else 0
        ) * sum_loss(loss_dict.loss_u)

        losses.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        self.optimizer.step()
        self.generator_optimizer.step()
        self.lr_scheduler.step()

        if dist_utils.is_distributed():
            self.model.module._momentum_update(self.cfg.solver.ema_momentum)
        else:
            self.model._momentum_update(self.cfg.solver.ema_momentum)

        return Dict(loss_dict=loss_dict, stat_dict=stat_dict)

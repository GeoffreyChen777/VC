from addict import Dict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from gvcore.utils.logger import logger
import gvcore.utils.distributed as dist_utils
from gvcore.optim.utils import sum_loss
from gvcore.dataset.build import build_dataloader
from gvcore.operators import OPERATOR_REGISTRY, GenericOpt
from gvcore.evaluator import EVALUATOR_REGISTRY
from gvcore.optim import make_lr_scheduler, make_optimizer
from gvcore.dataset.transforms import DataCopy, parse_transform_config

from model.segmentor.deeplabv3p_alt import DeeplabV3pAlt
from model.segmentor.ts import TSSegmentor


@OPERATOR_REGISTRY.register("teacher_student")
class TeacherStudentOpt(GenericOpt):
    def __init__(self, cfg):
        super(TeacherStudentOpt, self).__init__(cfg)

        self.train_loader_l = None
        self.train_loader_u = None

        self.evaluator_stu = None
        self.evaluator_tea = None

        self.weak_aug = DataCopy(parse_transform_config(cfg.data.weak_transforms))
        self.strong_aug = DataCopy(parse_transform_config(cfg.data.strong_transforms))

        self.burnin_step = cfg.solver.burnin_iter_num

    def build_model(self):
        model = TSSegmentor(self.cfg, DeeplabV3pAlt)
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
        self.lr_scheduler = make_lr_scheduler(self.cfg.solver.lr_scheduler, self.optimizer)

    def build_dataloader(self):
        self.train_loader_l = build_dataloader(self.cfg.data["train_l"], self.cfg.seed)
        self.train_loader_u = build_dataloader(self.cfg.data["train_u"], self.cfg.seed)
        self.test_loader = build_dataloader(self.cfg.data["test"], self.cfg.seed)
        logger.info(
            f"Dataset: Train Labeled[{len(self.train_loader_l)}], Train Unlabeled[{len(self.train_loader_u)}], Test[{len(self.test_loader)}]"
        )

    def train_pre_step(self) -> Dict:
        self.optimizer.zero_grad()

        if self.cur_step == self.burnin_step:
            logger.info(f"Initialize teacher...")
            if dist_utils.is_distributed():
                self.model.module._momentum_update(0)
            else:
                self.model._momentum_update(0)

        data_list_l = self.train_loader_l.get_batch("cuda")
        data_list_u = self.train_loader_u.get_batch("cuda")

        data_list_l_weak = self.weak_aug(data_list_l)
        data_list_u_weak = self.weak_aug(data_list_u)
        data_list_l_strong = self.strong_aug(data_list_l)
        data_list_u_strong = self.strong_aug(data_list_u)

        return Dict(
            data_list_l_weak=data_list_l_weak,
            data_list_u_weak=data_list_u_weak,
            data_list_l_strong=data_list_l_strong,
            data_list_u_strong=data_list_u_strong,
        )

    def train_run_step(self, model_inputs: Dict, **kwargs) -> Dict:
        loss_dict, stat_dict = self.model(**model_inputs)

        losses = sum_loss(loss_dict.loss_l) + (
            self.cfg.solver.u_loss_weight if self.cur_step > self.burnin_step else 0
        ) * sum_loss(loss_dict.loss_u)

        losses.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        self.optimizer.step()
        self.lr_scheduler.step()

        if dist_utils.is_distributed():
            self.model.module._momentum_update(self.cfg.solver.ema_momentum)
        else:
            self.model._momentum_update(self.cfg.solver.ema_momentum)

        return Dict(loss_dict=loss_dict, stat_dict=stat_dict)

    def train_after_step(self, run_outputs: Dict, **kwargs):
        loss_dict = run_outputs.loss_dict
        loss_dict_l = loss_dict.loss_l
        loss_dict_u = loss_dict.loss_u

        self.summary.update(loss_dict_l, namespace="l")
        self.summary.update(loss_dict_u, namespace="u")

        self.summary.update({"lr": self.lr_scheduler.get_last_lr()[0]})
        self.summary.update(run_outputs.stat_dict)

        if (self.cur_step + 1) % self.cfg.solver.ckp_interval == 0 and not self.cfg.dev_mode:
            self.save_ckp()
        if (self.cur_step + 1) % self.cfg.solver.get("eval_interval", self.cfg.solver.ckp_interval) == 0:
            self.validation()

        self.summary.summary()

    def build_evaluator(self):
        if self.evaluator_stu is None:
            self.evaluator_stu = EVALUATOR_REGISTRY[self.cfg.evaluator](
                self.cfg.model.num_classes, distributed=dist_utils.is_distributed()
            )
        if self.evaluator_tea is None:
            self.evaluator_tea = EVALUATOR_REGISTRY[self.cfg.evaluator](
                self.cfg.model.num_classes, distributed=dist_utils.is_distributed()
            )
        self.evaluator_stu.reset()
        self.evaluator_tea.reset()

    def run_test(self) -> Dict:
        logger.info("Start testing...")

        while True:
            model_inputs = self.test_pre_step()
            if model_inputs is None:
                break
            self.test_run_step(model_inputs)

        test_results = Dict()
        test_results.stu = self.evaluator_stu.evaluate()
        test_results.tea = self.evaluator_tea.evaluate()

        return test_results

    def test_run_step(self, model_inputs: Dict, **kwargs):
        pred_list_stu, pred_list_tea = self.model(**model_inputs)

        for data, pred_stu, pred_tea in zip(model_inputs.data_list, pred_list_stu, pred_list_tea):
            self.evaluator_stu.process(pred_stu, data.label)
            self.evaluator_tea.process(pred_tea, data.label)

    @torch.no_grad()
    def validation(self):
        self.summary.add_metrics(
            window_size=1,
            log_interval=self.cfg.solver.get("eval_interval", self.cfg.solver.ckp_interval),
            namespace="val_tea",
            printable=False,
        )
        self.summary.add_metrics(
            window_size=1,
            log_interval=self.cfg.solver.get("eval_interval", self.cfg.solver.ckp_interval),
            namespace="val_stu",
            printable=False,
        )
        self.build_evaluator()

        self.model.eval()

        test_results = self.run_test()
        self.summary.update(test_results.stu, namespace="val_stu")
        self.summary.update(test_results.tea, namespace="val_tea")

        self.model.train()

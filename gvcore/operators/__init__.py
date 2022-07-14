import torch
import numpy as np
import random

from gvcore.dataset.build import build_dataloader
from gvcore.model.utils import get_default_optimizer_params
from gvcore.utils.checkpoint import Checkpointer
from gvcore.utils.logger import GenericSummary, logger
from gvcore.utils.config import config_to_string
from gvcore.utils.lr_scheduler import MultiStepLR
from gvcore.utils.registry import Registry
from gvcore.evaluator import EVALUATOR_REGISTRY

OPERATOR_REGISTRY = Registry()


class GenericOpt:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_random_seed(cfg.seed)
        self.device = torch.device("cuda")

        # Basic components for most network
        self.model = None
        self.optimizer = None
        self.lr_scheduler = None

        self.train_loader = None
        self.test_loader = None

        self.evaluator = None

        self.checkpointer = Checkpointer(cfg)
        self.cur_step = 0
        self.iter_num = cfg.solver.iter_num

        self.summary = GenericSummary(cfg=cfg, iter_num=cfg.solver.iter_num)

    @staticmethod
    def set_random_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def build_model(self):
        raise NotImplementedError

    def build_optimizer(self):
        params = get_default_optimizer_params(
            self.model, base_lr=self.cfg.solver.lr, weight_decay=self.cfg.solver.weight_decay, weight_decay_norm=0
        )
        self.optimizer = torch.optim.SGD(
            params, lr=self.cfg.solver.lr, momentum=self.cfg.solver.momentum, weight_decay=self.cfg.solver.weight_decay,
        )
        self.lr_scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.cfg.solver.lr_steps,
            gamma=self.cfg.solver.lr_gamma,
            warmup=self.cfg.solver.warmup,
            warmup_step=self.cfg.solver.warmup_steps,
            warmup_gamma=self.cfg.solver.warmup_gamma,
        )

    def build_dataloader(self):
        self.train_loader = build_dataloader(self.cfg, "train")
        self.test_loader = build_dataloader(self.cfg, "test")
        logger.info(f"Dataset: Train[{len(self.train_loader)}], Test[{len(self.test_loader)}]")

    def save_ckp(self, name=None):
        self.checkpointer.save(
            name=name if name is not None else self.cur_step + 1,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            step=self.cur_step,
        )

    def resume_ckp(self, test_mode=False):
        if self.cfg.resume is None:
            return
        logger.info(f"Resuming from ckp: {self.cfg.resume}")

        if not test_mode:
            assert (
                self.model is not None and self.optimizer is not None and self.lr_scheduler is not None
            ), f"Checkpoint resuming should be operated after model, optimizer and lr scheduler building: Model: {type(self.model)}, Optimizer: {type(self.model)}, LR sch: {type(self.lr_scheduler)}"

            if self.cfg.solver.warmup:
                self.cur_step = self.checkpointer.load(self.cfg.resume, model=self.model, optimizer=self.optimizer)
            else:
                self.cur_step = self.checkpointer.load(
                    self.cfg.resume, model=self.model, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler,
                )
            self.summary.step = self.cur_step
        else:
            self.cur_step = self.checkpointer.load(self.cfg.resume, model=self.model)

    """
    train:
     |   1. before_train:
     |   2. run_train:
     |     for i < range(iter_num):
     |       a. train_run_step:
     |       b. train_after_step:
     v   3. after_train
    """

    def train(self):
        self.before_train()
        self.run_train()
        self.after_train()

    def before_train(self):
        cfg = self.cfg
        logger.info("Prepare for training...")
        # if not cfg.dev_mode:
        logger.info("\n" + config_to_string(cfg))

        self.build_model()
        self.build_optimizer()
        self.build_dataloader()
        self.resume_ckp()

        self.model.train()

    def run_train(self):
        logger.info(f"Start training...")

        for cur_step in range(self.cur_step, self.iter_num):
            self.cur_step = cur_step
            self.train_run_step()
            self.train_after_step()

    def train_run_step(self):
        raise NotImplementedError

    def train_after_step(self):
        if (self.cur_step + 1) % self.cfg.solver.ckp_interval == 0 and not self.cfg.dev_mode:
            self.save_ckp()
        if (self.cur_step + 1) % self.cfg.solver.ckp_interval == 0:
            self.validation()
        self.summary.summary()

    def after_train(self):
        self.save_ckp(name="final")
        self.validation()

    """
    test:
     |   1. before_test:
     |   2. run_test:
     |     while True:
     |       test_run_step:
     v   3. after_test: Optional
    """

    def build_evaluator(self):
        cfg = self.cfg
        if self.evaluator is None:
            self.evaluator = EVALUATOR_REGISTRY[cfg.evaluator](cfg.data.test.json_file, distributed=cfg.distributed.use)
        self.evaluator.reset()

    def test(self):
        self.before_test()
        self.run_test()

    def before_test(self):
        cfg = self.cfg
        assert cfg.resume is not None, "Please use --resume to set trained model path for testing."
        logger.info("Prepare for testing...")

        self.build_model()
        self.build_dataloader()
        self.build_evaluator()

        self.resume_ckp(test_mode=True)

        self.model.eval()

    @torch.no_grad()
    def run_test(self):
        logger.info("Start testing...")

        while True:
            finished = self.test_run_step()
            if finished:
                break
        return self.evaluator.evaluate()

    def test_run_step(self, *args, **kwargs):
        raise NotImplementedError

    def validation(self):
        self.summary.add_metrics(
            window_size=1, log_interval=self.cfg.solver.ckp_interval, namespace="val", printable=False
        )
        self.build_evaluator()

        self.model.eval()
        stat = self.run_test()
        self.summary.update({"mAP": stat[0], "mAR": stat[8]}, namespace="val")
        self.model.train()

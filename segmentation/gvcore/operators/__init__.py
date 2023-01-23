import torch
import numpy as np
import random
from typing import Optional, Tuple
from addict import Dict

from gvcore.dataset.build import build_dataloader
from gvcore.optim import make_lr_scheduler, make_optimizer
from gvcore.optim.utils import sum_loss
from gvcore.utils.checkpoint import Checkpointer
from gvcore.utils.logger import GenericSummary, logger
from gvcore.utils.config import config_to_string
from gvcore.utils.distributed import is_distributed
from gvcore.utils.registry import Registry
from gvcore.evaluator import EVALUATOR_REGISTRY

OPERATOR_REGISTRY = Registry()


class GenericOpt:
    def __init__(self, cfg):
        self.cfg = cfg
        self.set_random_seed(cfg.seed)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        # Basic components for most models
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
        self.optimizer = make_optimizer(self.cfg.solver.optimizer, self.model)
        self.lr_scheduler = make_lr_scheduler(self.cfg.solver.lr_scheduler, self.optimizer)

    def build_dataloader(self):
        self.train_loader = build_dataloader(self.cfg.data["train"], self.cfg.seed)
        self.test_loader = build_dataloader(self.cfg.data["test"], self.cfg.seed)
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
     |       a. train_pre_step:
     |       b. train_run_step:
     |       c. train_after_step:
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
            model_inputs = self.train_pre_step()
            run_outputs = self.train_run_step(model_inputs)
            self.train_after_step(run_outputs)

    def train_pre_step(self) -> Dict:
        """
        Train pre step:
            1. Zero gradients
            2. Get data
        """
        self.optimizer.zero_grad()
        data_list = self.train_loader.get_batch(self.device)
        return Dict(data_list=data_list)

    def train_run_step(self, model_inputs: Dict, **kwargs) -> Dict:
        """
        Train run step:
            1. Forward model
            2. Compute loss
            3. Backward and update
        """
        loss_dict = self.model(**model_inputs)
        losses = sum_loss(loss_dict)
        losses.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        return Dict(loss_dict=loss_dict)

    def train_after_step(self, run_outputs: Dict, **kwargs):
        """
        Train after step:
            1. Do logging stuff.
            2. Save checkpoint.
            3. Validation.
        """
        if (self.cur_step + 1) % self.cfg.solver.ckp_interval == 0 and not self.cfg.dev_mode:
            self.save_ckp()
        if (self.cur_step + 1) % self.cfg.solver.get("eval_interval", self.cfg.solver.ckp_interval) == 0:
            self.validation()

        loss_dict = run_outputs.loss_dict
        loss_dict.update({"lr": self.lr_scheduler.get_last_lr()[0]})
        self.summary.update(loss_dict)
        self.summary.summary()

    def after_train(self):
        self.save_ckp(name="final")
        self.validation()

    """
    test:
     |   1. before_test:
     |   2. run_test:
     |     while True:
     |       a. test_pre_step:
     |       b. test_run_step:
     |       c. test_after_step: Optinal
     v   3. after_test: Optional
    """

    def build_evaluator(self):
        cfg = self.cfg.evaluator
        if self.evaluator is None:
            type = cfg.type
            cfg.pop("type")
            self.evaluator = EVALUATOR_REGISTRY[type](**cfg, distributed=is_distributed())
        self.evaluator.reset()

    @torch.no_grad()
    def test(self):
        self.before_test()
        self.run_test()

    def before_test(self):
        cfg = self.cfg
        if cfg.resume is None:
            logger.warning("Please use --resume to set trained model path for testing.")
        logger.info("Prepare for testing...")

        self.build_model()
        self.build_dataloader()
        self.build_evaluator()

        self.resume_ckp(test_mode=True)

        self.model.eval()

    def run_test(self):
        logger.info("Start testing...")

        while True:
            model_inputs = self.test_pre_step()
            if model_inputs is None:
                break
            run_outputs = self.test_run_step(model_inputs)
            model_inputs, run_outputs = self.test_after_step(model_inputs, run_outputs)

            self.evaluate_process(model_inputs, run_outputs)
        return self.evaluator.evaluate()

    def test_pre_step(self) -> Optional[Dict]:
        data_list = self.test_loader.get_batch(self.device)
        if data_list is None:
            return None
        return Dict(data_list=data_list)

    def test_run_step(self, model_inputs: Dict, **kwargs) -> Dict:
        run_outputs = self.model(**model_inputs)
        return run_outputs

    def test_after_step(self, model_inputs: Dict, run_outputs: Dict, **kwargs) -> Tuple[Dict, Dict]:
        return model_inputs, run_outputs

    def evaluate_process(self, model_inputs: Dict, run_outputs: Dict, **kwargs):
        for data, pred in zip(model_inputs.data_list, run_outputs.pred_list):
            self.evaluator.process(pred, data.label)

    """
    validation
    """

    @torch.no_grad()
    def validation(self):
        self.summary.add_metrics(
            window_size=1, log_interval=self.cfg.solver.get("eval_interval", self.cfg.solver.ckp_interval), namespace="val", printable=False
        )
        self.build_evaluator()

        self.model.eval()
        stat = self.run_test()
        self.summary.update(stat, namespace="val")
        self.model.train()

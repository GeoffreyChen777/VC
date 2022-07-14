import torch
from torch.nn.parallel import DistributedDataParallel
from torchvision.ops import clip_boxes_to_image

from gvcore.utils.logger import logger
import gvcore.utils.distributed as dist_utils
from gvcore.utils.misc import sum_loss
from gvcore.dataset.build import build_dataloader
from gvcore.operators import OPERATOR_REGISTRY, GenericOpt

from model.detector.ts import TSDetector
from model.detector.unbiased_teacher import UBTDetector


@OPERATOR_REGISTRY.register("unbiased_teacher")
class UnbiasedTeacherOpt(GenericOpt):
    def __init__(self, cfg):
        super(UnbiasedTeacherOpt, self).__init__(cfg)

        self.train_loader_l = None
        self.train_loader_u = None

        self.burnin_step = cfg.solver.burnin_iter_num

    def build_model(self):
        model = TSDetector(self.cfg, UBTDetector)
        model.to(self.device)
        if dist_utils.is_distributed():
            model = DistributedDataParallel(model)
        self.model = model

    def build_dataloader(self):
        self.train_loader_l = build_dataloader(self.cfg, "train_l")
        self.train_loader_u = build_dataloader(self.cfg, "train_u")
        self.test_loader = build_dataloader(self.cfg, "test")
        logger.info(
            f"Dataset: Train Labeled[{len(self.train_loader_l)}], Train Unlabeled[{len(self.train_loader_u)}], Test[{len(self.test_loader)}]"
        )

    def train_run_step(self):
        self.optimizer.zero_grad()

        data_list_l = self.train_loader_l.get_batch("cuda")
        data_list_u = self.train_loader_u.get_batch("cuda")

        # 1. Train with labeled data
        loss_dict_l = self.model(data_list_l, stage="train_l")
        losses_l = sum_loss(loss_dict_l)
        self.summary.update(loss_dict_l, namespace="l")

        # 2. Initialize the teacher model
        if self.cur_step == self.burnin_step:
            if dist_utils.is_distributed():
                self.model.module._momentum_update(m=0.0)
            else:
                self.model._momentum_update(m=0.0)
            self.save_ckp()
            self.validation()

        # 3. Train with unlabeled data
        if self.cur_step >= self.burnin_step:
            loss_dict_u = self.model(data_list_u, stage="train_u")
            losses_u = sum_loss(loss_dict_u)
            self.summary.update(loss_dict_u, namespace="u")
        else:
            losses_u = torch.tensor(0.0, device=self.device)

        losses = losses_l + self.cfg.solver.u_loss_weight * losses_u

        losses.backward()
        self.optimizer.step()

        self.summary.update({"lr": self.lr_scheduler.get_last_lr()[0]})

        self.lr_scheduler.step()

    @torch.no_grad()
    def run_test(self):
        logger.info("Start testing student...")
        stats = []
        while True:
            finished = self.test_run_step(selector="student")
            if finished:
                break
        stu_stat = self.evaluator.evaluate()
        if stu_stat is not None:
            stu_mAP = stu_stat[0]
            stu_mAR = stu_stat[8]
            stats.append(stu_mAP)
            stats.append(stu_mAR)
        self.evaluator.reset()

        logger.info("Start testing teacher...")
        while True:
            finished = self.test_run_step(selector="teacher")
            if finished:
                break
        tea_stat = self.evaluator.evaluate()
        if tea_stat is not None:
            tea_mAP = tea_stat[0]
            tea_mAR = tea_stat[8]
            stats.append(tea_mAP)
            stats.append(tea_mAR)
        return stats

    def test_run_step(self, selector="model"):
        data_list = self.test_loader.get_batch("cuda")
        if data_list is None:
            return True
        detected_bboxes = self.model(data_list, selector=selector)

        for data, detected_bbox in zip(data_list, detected_bboxes):
            meta = data.meta
            scale_x, scale_y = (
                1.0 * meta.ori_size[1] / meta.cur_size[1],
                1.0 * meta.ori_size[0] / meta.cur_size[0],
            )
            detected_bbox[:, [0, 2]] *= scale_x
            detected_bbox[:, [1, 3]] *= scale_y
            detected_bbox[:, :4] = clip_boxes_to_image(detected_bbox[:, :4], meta.ori_size)

            self.evaluator.process(data.meta.id, detected_bbox)
        return False

    def validation(self):
        self.summary.add_metrics(
            window_size=1, log_interval=self.cfg.solver.ckp_interval, namespace="val_tea", printable=False
        )
        self.summary.add_metrics(
            window_size=1, log_interval=self.cfg.solver.ckp_interval, namespace="val_stu", printable=False
        )
        self.build_evaluator()

        self.model.eval()

        stats = self.run_test()
        if len(stats) >= 4:
            self.summary.update({"mAP": stats[0], "mAR": stats[1]}, namespace="val_stu")
            self.summary.update({"mAP": stats[2], "mAR": stats[3]}, namespace="val_tea")

        self.model.train()

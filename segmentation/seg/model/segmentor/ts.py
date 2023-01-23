from typing import Callable, Optional
from addict import Dict
import torch
import torch.nn.functional as F

from gvcore.utils.types import TDataList
from gvcore.utils.misc import sharpen
from gvcore.model import GenericModule

from evaluator.segmentation import SegmentationEvaluator


class TSSegmentor(GenericModule):
    """
    Teacher-Student segmentor for semi-supervised learning.
    """

    def __init__(self, cfg, segmentor_class: Callable[..., GenericModule]):
        super(TSSegmentor, self).__init__(cfg)

        # Components ========
        self.segmentor = segmentor_class(cfg)
        self.ema_segmentor = segmentor_class(cfg)
        self.ema_segmentor.eval()

        # Initialize ========
        for param_q, param_k in zip(self.segmentor.parameters(), self.ema_segmentor.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # Buffer ========
        for m in self.ema_segmentor.modules():
            if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.SyncBatchNorm)):
                m.momentum = 0.9

        self.bn_statistics_pl = Dict()
        self.bn_statistics_eval = Dict()
        self.bn_statistics_mode = "eval"

        for name, buffer in self.ema_segmentor.named_buffers():
            self.bn_statistics_pl[name] = buffer.clone()
            self.bn_statistics_eval[name] = buffer.clone()

        # Tools ========
        self.tranining_evaluator = SegmentationEvaluator(
            num_classes=cfg.model.num_classes, distributed=cfg.num_gpus > 1, mode="window_avg", window_size=100
        )

    def to(self, device):
        super(TSSegmentor, self).to(device)
        for name, _ in self.ema_segmentor.named_buffers():
            self.bn_statistics_eval[name] = self.bn_statistics_eval[name].to(device)
            self.bn_statistics_pl[name] = self.bn_statistics_pl[name].to(device)

    def train(self, mode=True):
        self.training = mode
        self.segmentor.train(mode)
        self.ema_segmentor.eval()

    def switch_to_bn_statistics_pl(self):
        if self.bn_statistics_mode != "pl":
            for name, buffer in self.ema_segmentor.named_buffers():
                self.bn_statistics_eval[name] = buffer.clone()
                buffer.data = self.bn_statistics_pl[name]
            self.bn_statistics_mode = "pl"

    def switch_to_bn_statistics_eval(self):
        if self.bn_statistics_mode != "eval":
            for name, buffer in self.ema_segmentor.named_buffers():
                self.bn_statistics_pl[name] = buffer.clone()
                buffer.data = self.bn_statistics_eval[name]
            self.bn_statistics_mode = "eval"

    def forward_generic(self, *args, **kwargs):
        selector = kwargs.pop("selector")
        if selector == "student":
            return self.segmentor.forward_generic(*args, **kwargs)
        elif selector == "teacher":
            return self.ema_segmentor.forward_generic(*args, **kwargs)

    def forward_train(
        self,
        data_list_l_weak: TDataList,
        data_list_u_weak: TDataList,
        data_list_l_strong: TDataList,
        data_list_u_strong: TDataList,
    ):
        logits_l = self.forward_generic(data_list_l_strong, selector="student")
        # 1. Train with labeled data
        labels_l = torch.cat([data.label for data in data_list_l_weak], dim=0)
        loss_dict_l = self.get_losses(logits_l, labels_l, gt_labels=labels_l, labelled=True)

        # 2. Pseudo labeling and Consistency training with labeled and unlabeled data
        pseudo_labels_u = self._pseudo_labeling(data_list_u_weak)
        labels_u = torch.cat([data.label for data in data_list_u_weak], dim=0)
        logits_u = self.forward_generic(data_list_u_strong, selector="student")
        loss_dict_u = self.get_losses(logits_u, pseudo_labels_u, gt_labels=labels_u, labelled=False)

        loss_dict = Dict()
        loss_dict.loss_l = loss_dict_l
        loss_dict.loss_u = loss_dict_u

        # 3. Calculate mIoU of pseudo labels
        stat_dict = self._pseudo_labeling_statistics(pseudo_labels_u, labels_u)

        return loss_dict, stat_dict

    def get_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        labelled: bool = True,
    ):
        loss_dict = Dict()
        if labelled:
            loss = F.cross_entropy(logits, labels.long(), ignore_index=255)
        else:
            pseudo_probs_u, pseudo_preds_u = torch.max(labels, dim=1)
            mask = torch.ge(pseudo_probs_u, self.cfg.model.teacher.score_threshold)
            valid_mask = torch.ne(gt_labels, 255) & mask

            loss = F.cross_entropy(logits, pseudo_preds_u, reduction="none")
            loss = loss * valid_mask.float()

            loss = loss.sum() / (valid_mask.sum() + 1e-10)
        loss_dict.loss = loss

        return loss_dict

    @torch.no_grad()
    def _pseudo_labeling(self, data_list: TDataList, sharpen_factor: float = 0.5) -> torch.Tensor:
        # Update momentum batch statistic pl
        self.switch_to_bn_statistics_pl()
        self.ema_segmentor.train()
        self.forward_generic(data_list, selector="teacher")
        self.ema_segmentor.eval()

        logits = self.forward_generic(data_list, selector="teacher")
        probs = F.softmax(logits, dim=1)
        if sharpen_factor > 0:
            probs = sharpen(probs, sharpen_factor)

        self.switch_to_bn_statistics_eval()
        return probs.detach()

    @torch.no_grad()
    def _pseudo_labeling_statistics(self, pseudo_labels: torch.Tensor, gt_labels: torch.Tensor) -> Dict:
        if pseudo_labels.dim() == 4:
            pseudo_labels = torch.argmax(pseudo_labels, dim=1)

        self.tranining_evaluator.process(pseudo_labels, gt_labels)
        metrics = self.tranining_evaluator.calculate()

        return Dict(PL_mIoU=metrics["mIoU"])

    @torch.no_grad()
    def _momentum_update(self, m: float):
        """
        Momentum update of the key encoder
        """
        for buffer_q, buffer_k in zip(self.segmentor.buffers(), self.ema_segmentor.buffers()):
            buffer_k.data = buffer_k.data * m + buffer_q.data * (1.0 - m)

        for param_q, param_k in zip(self.segmentor.parameters(), self.ema_segmentor.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

    @torch.no_grad()
    def forward_eval(self, data_list: TDataList):
        # 1. Test student
        logits_stu = self.forward_generic(data_list, selector="student")
        probs_stu = F.softmax(logits_stu, dim=1)
        preds_stu = torch.argmax(probs_stu, dim=1)

        # 2. Test teacher
        logits_tea = self.forward_generic(data_list, selector="teacher")
        probs_tea = F.softmax(logits_tea, dim=1)
        preds_tea = torch.argmax(probs_tea, dim=1)

        return preds_stu, preds_tea

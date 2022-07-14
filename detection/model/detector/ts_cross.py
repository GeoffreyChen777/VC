import torch

from gvcore.utils.types import TDataList
from gvcore.utils.misc import merge_loss_dict, split_ab
from gvcore.model import GenericModule

from dataset.transforms import DataCopy, parse_transform_config


class TSCrossDetector(GenericModule):
    """
    Teacher-Student detector for semi-supervised learning.
    """

    def __init__(self, cfg, detector_class: GenericModule):
        super(TSCrossDetector, self).__init__(cfg)

        # Components ========
        self.detector_a = detector_class(cfg)
        self.ema_detector_a = detector_class(cfg)
        self.detector_b = detector_class(cfg)
        self.ema_detector_b = detector_class(cfg)

        # Tools =============
        self.weak_aug = DataCopy(parse_transform_config(cfg.data.weak_transforms))
        self.strong_aug = DataCopy(parse_transform_config(cfg.data.strong_transforms))

        # Initialize ========
        for param_q, param_k in zip(self.detector_a.parameters(), self.ema_detector_a.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.ema_detector_a.freeze(True)
        self.ema_detector_a.eval()

        for param_q, param_k in zip(self.detector_b.parameters(), self.ema_detector_b.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.ema_detector_b.freeze(True)
        self.ema_detector_b.eval()

        # Params ==============
        self.m = cfg.solver.ema_momentum
        self.pseudo_t = cfg.model.teacher.score_threshold

    def train(self, mode: bool = True):
        self.training = mode
        self.detector_a.train(mode)
        self.detector_b.train(mode)

    def forward(self, *args, **kwargs):
        if "stage" not in kwargs:
            stage = "train_l" if self.training else "eval"
        else:
            stage = kwargs.pop("stage")

        if stage == "train_l":
            return self.forward_train_l(*args, **kwargs)
        elif stage == "train_u":
            return self.forward_train_u(*args, **kwargs)
        elif stage == "eval":
            return self.forward_eval(*args, **kwargs)

    def forward_train_l(self, data_list_l):
        weak_data_list_l = self.weak_aug(data_list_l)
        strong_data_list_l = self.strong_aug(data_list_l)
        weak_data_list_l_a, weak_data_list_l_b = split_ab(weak_data_list_l)
        strong_data_list_l_a, strong_data_list_l_b = split_ab(strong_data_list_l)

        data_list_l_a = weak_data_list_l_a + strong_data_list_l_a
        data_list_l_b = weak_data_list_l_b + strong_data_list_l_b

        loss_dict_l_a = self.detector_a(data_list_l_a, labeled=True)
        loss_dict_l_b = self.detector_b(data_list_l_b, labeled=True)

        loss_dict_l = merge_loss_dict(loss_dict_l_a, loss_dict_l_b)
        return loss_dict_l

    def forward_train_u(self, data_list_u: TDataList):
        self._momentum_update(m=self.m)

        weak_data_list_u = self.weak_aug(data_list_u)
        strong_data_list_u = self.strong_aug(data_list_u)
        weak_data_list_u_a, weak_data_list_u_b = split_ab(weak_data_list_u)
        strong_data_list_u_a, strong_data_list_u_b = split_ab(strong_data_list_u)

        pseudo_labels_a = self.ema_detector_b(weak_data_list_u_a)
        for data, pseudo_label in zip(strong_data_list_u_a, pseudo_labels_a):
            select = pseudo_label[:, 5] > self.pseudo_t
            data.label = torch.narrow(pseudo_label[select], dim=1, start=0, length=5)
        pseudo_labels_b = self.ema_detector_a(weak_data_list_u_b)
        for data, pseudo_label in zip(strong_data_list_u_b, pseudo_labels_b):
            select = pseudo_label[:, 5] > self.pseudo_t
            data.label = torch.narrow(pseudo_label[select], dim=1, start=0, length=5)

        loss_dict_u_a = self.detector_a(strong_data_list_u_a, labeled=False)
        loss_dict_u_b = self.detector_b(strong_data_list_u_b, labeled=False)

        loss_dict_u = merge_loss_dict(loss_dict_u_a, loss_dict_u_b)

        return loss_dict_u

    def forward_eval(self, *args, **kwargs):
        selector = kwargs.pop("selector")
        if selector == "student":
            return self.detector_a(*args, **kwargs)
        elif selector == "teacher":
            return self.ema_detector_a(*args, **kwargs)

    @torch.no_grad()
    def _momentum_update(self, m: float):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.detector_a.parameters(), self.ema_detector_a.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)
        for param_q, param_k in zip(self.detector_b.parameters(), self.ema_detector_b.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

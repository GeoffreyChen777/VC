import torch

from gvcore.utils.types import TDataList
from gvcore.model import GenericModule

from dataset.transforms import DataCopy, parse_transform_config


class TSDetector(GenericModule):
    """
    Teacher-Student detector for semi-supervised learning.
    """

    def __init__(self, cfg, detector_class: GenericModule):
        super(TSDetector, self).__init__(cfg)

        # Components ========
        self.detector = detector_class(cfg)
        self.ema_detector = detector_class(cfg)

        # Tools =============
        self.weak_aug = DataCopy(parse_transform_config(cfg.data.weak_transforms))
        self.strong_aug = DataCopy(parse_transform_config(cfg.data.strong_transforms))

        # Initialize ========
        for param_q, param_k in zip(self.detector.parameters(), self.ema_detector.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        self.ema_detector.freeze(True)
        self.ema_detector.eval()

        # Params ==============
        self.m = cfg.solver.ema_momentum
        self.pseudo_t = cfg.model.teacher.score_threshold

    def train(self, mode: bool = True):
        self.training = mode
        self.detector.train(mode)

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

    def forward_train_l(self, data_list_l: TDataList):
        weak_data_list_l = self.weak_aug(data_list_l)
        strong_data_list_l = self.strong_aug(data_list_l)
        data_list_l = weak_data_list_l + strong_data_list_l

        loss_dict_l = self.detector(data_list_l, labeled=True)

        return loss_dict_l

    def forward_train_u(self, data_list_u: TDataList):
        self._momentum_update(m=self.m)

        weak_data_list_u = self.weak_aug(data_list_u)
        strong_data_list_u = self.strong_aug(data_list_u)

        pseudo_labels = self.ema_detector(weak_data_list_u)
        for data, pseudo_label in zip(strong_data_list_u, pseudo_labels):
            select = pseudo_label[:, 5] > self.pseudo_t
            data.label = torch.narrow(pseudo_label[select], dim=1, start=0, length=5)

        loss_dict_u = self.detector(strong_data_list_u, labeled=False)

        return loss_dict_u

    def forward_eval(self, *args, **kwargs):
        selector = kwargs.pop("selector")
        if selector == "student":
            return self.detector(*args, **kwargs)
        elif selector == "teacher":
            return self.ema_detector(*args, **kwargs)

    @torch.no_grad()
    def _momentum_update(self, m: float):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.detector.parameters(), self.ema_detector.parameters()):
            param_k.data = param_k.data * m + param_q.data * (1.0 - m)

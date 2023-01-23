from typing import Dict, Literal
import torch
from tabulate import tabulate
import torch.distributed as dist


from gvcore.utils.logger import logger
import gvcore.utils.distributed as dist_utils
from gvcore.utils.structure import TensorList
from gvcore.evaluator import EVALUATOR_REGISTRY

from evaluator.api import intersect_and_union


@EVALUATOR_REGISTRY.register("segmentation")
class SegmentationEvaluator:
    def __init__(
        self, num_classes, distributed=False, mode: Literal["onetime", "window_avg"] = "onetime", window_size=0
    ):
        self.num_classes = num_classes
        self._distributed = distributed
        self._mode = mode
        self.window_size = window_size

        if self._mode == "onetime":
            self._total_area_intersect = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
            self._total_area_union = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
            self._total_area_pred = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
            self._total_area_label = torch.zeros((self.num_classes,), dtype=torch.float64, device="cuda")
        elif self._mode == "window_avg":
            self._total_area_intersect = TensorList(
                (window_size, num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._total_area_union = TensorList(
                (window_size, num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._total_area_pred = TensorList(
                (window_size, num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._total_area_label = TensorList(
                (window_size, num_classes), init_value=0, dtype=torch.float64, device="cuda", all_gather=False
            )
            self._idx = torch.tensor([0], dtype=torch.int64, device="cuda")
        else:
            raise ValueError("Invalid mode: {}".format(self._mode))

    def reset(self):
        self._total_area_intersect.fill_(0)
        self._total_area_union.fill_(0)
        self._total_area_pred.fill_(0)
        self._total_area_label.fill_(0)

    def process(
        self,
        pred: torch.Tensor,
        label: torch.Tensor,
        ignore_index: int = 255,
        label_map: Dict = {},
        reduce_zero_label: bool = False,
    ):
        area_intersect, area_union, area_pred, area_label = intersect_and_union(
            pred, label, self.num_classes, ignore_index, label_map, reduce_zero_label
        )

        if self._mode == "onetime":
            self._total_area_intersect += area_intersect
            self._total_area_union += area_union
            self._total_area_pred += area_pred
            self._total_area_label += area_label
        elif self._mode == "window_avg":
            self._total_area_intersect[self._idx] = area_intersect.unsqueeze(0)
            self._total_area_union[self._idx] = area_union.unsqueeze(0)
            self._total_area_pred[self._idx] = area_pred.unsqueeze(0)
            self._total_area_label[self._idx] = area_label.unsqueeze(0)
            self._idx = (self._idx + 1) % self.window_size

    def calculate(self):
        if self._mode == "onetime":
            total_area_intersect = self._total_area_intersect.clone()
            total_area_union = self._total_area_union.clone()
            total_area_pred = self._total_area_pred.clone()
            total_area_label = self._total_area_label.clone()
        elif self._mode == "window_avg":
            total_area_intersect = self._total_area_intersect._tensor.clone()
            total_area_union = self._total_area_union._tensor.clone()
            total_area_pred = self._total_area_pred._tensor.clone()
            total_area_label = self._total_area_label._tensor.clone()

        if self._distributed:
            dist_utils.synchronize()
            dist.all_reduce(total_area_intersect)
            dist.all_reduce(total_area_union)
            dist.all_reduce(total_area_pred)
            dist.all_reduce(total_area_label)

        if self._mode == "window_avg":
            total_area_intersect = total_area_intersect.sum(dim=0)
            total_area_union = total_area_union.sum(dim=0)
            total_area_pred = total_area_pred.sum(dim=0)
            total_area_label = total_area_label.sum(dim=0)

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        metrics = {"aAcc": all_acc}
        iou = total_area_intersect / total_area_union
        acc = total_area_intersect / total_area_label
        metrics["mIoU"] = iou.mean()
        metrics["mAcc"] = acc.mean()

        return metrics

    def evaluate(self):
        metrics = self.calculate()
        logger.info(
            "Evaluation results: \n{}".format(
                tabulate([[metric, "{:.4f}".format(value)] for metric, value in metrics.items()], tablefmt="github",)
            )
        )
        return metrics

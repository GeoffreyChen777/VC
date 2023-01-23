from utils.box import pairwise_iou
import torch
from gvcore.utils.types import TTensor, TTensorList, TTensorTuple


def single_bbox_quality(pred: TTensor, label: TTensor, iou_t: int = 0.0) -> TTensorTuple:
    """Evaluate predicted bbox quality, return precision, recall, mean iou.
    """
    if pred.numel() > 0:
        iou = pairwise_iou(torch.narrow(pred, dim=1, start=0, length=4), torch.narrow(label, dim=1, start=0, length=4))

        pred_cls = pred[:, 4]
        label_cls = label[:, 4]

        # ======================
        # Precision
        max_iou, max_idx = torch.max(iou, dim=1)

        mean_iou = max_iou.mean()

        matched_cls = label_cls[max_idx]
        matched_cls[max_iou < iou_t] = -1

        correct_flag = torch.eq(pred_cls, matched_cls)
        precision = correct_flag.float().sum() / matched_cls.numel()

        # ======================
        # Recall
        max_iou, max_idx = torch.max(iou, dim=0)

        matched_cls = pred_cls[max_idx]
        matched_cls[max_iou < iou_t] = -1

        correct_flag = torch.eq(label_cls, matched_cls)
        recall = correct_flag.float().sum() / label_cls.numel()

        return precision, recall, mean_iou
    else:
        return (
            torch.tensor([0,], device=pred.device),
            torch.tensor([0,], device=pred.device),
            torch.tensor([0,], device=pred.device),
        )


def bbox_quality(pred_list: TTensorList, label_list: TTensorList, iou_t: int = 0.0) -> TTensorTuple:
    """Evaluate predicted bbox quality, return precision, recall, mean iou.
    """
    assert len(pred_list) == len(label_list), "Number of results and labels mismatch."
    assert len(pred_list) != 0, "Number of predictions cannot be zero."

    precision_list, recall_list, meaniou_list = [], [], []

    for pred, label in zip(pred_list, label_list):
        p, r, miou = single_bbox_quality(pred, label, iou_t)
        precision_list.append(p)
        recall_list.append(r)
        meaniou_list.append(miou)

    return (
        sum(precision_list) / len(precision_list),
        sum(recall_list) / len(recall_list),
        sum(meaniou_list) / len(meaniou_list),
    )

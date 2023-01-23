import torch
from torchvision.ops import clip_boxes_to_image


def pairwise_intersection(boxes1, boxes2) -> torch.Tensor:
    """
    Returns:
        Tensor: intersection, sized [N,M].
    """
    width_height = torch.min(boxes1[:, None, 2:], boxes2[:, 2:]) - torch.max(
        boxes1[:, None, :2], boxes2[:, :2]
    )  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    intersection = width_height.prod(dim=2)  # [N,M]
    return intersection


def pairwise_iou(boxes1, boxes2) -> torch.Tensor:
    """
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    inter = pairwise_intersection(boxes1, boxes2)

    # handle empty boxes
    iou = torch.where(
        inter > 0, inter / (area1[:, None] + area2 - inter), torch.zeros(1, dtype=inter.dtype, device=inter.device)
    )
    return iou


def cxcywh2ltxywh(box):
    box = box.clone()
    box[:, :2] -= box[:, 2:4] / 2
    return box


def xyxy2ltxywh(box):
    box = box.clone()
    box[:, 2:4] = box[:, 2:4] - box[:, :2]
    return box


def xyxy2cxcywh(box):
    box = box.clone()
    box[:, 2:4] = box[:, 2:4] - box[:, :2]
    box[:, 0:2] += box[:, 2:4] / 2
    return box


def nonempty(box, threshold=1e-5):
    widths = box[:, 2] - box[:, 0]
    heights = box[:, 3] - box[:, 1]
    keep = (widths > threshold) & (heights > threshold)
    return keep


def rescale(box, meta):
    scale_x, scale_y = (
        1.0 * meta.ori_size[1] / (meta.cur_size[1] - meta.pad_size[2]),
        1.0 * meta.ori_size[0] / (meta.cur_size[0] - meta.pad_size[3]),
    )
    box[:, [0, 2]] *= scale_x
    box[:, [1, 3]] *= scale_y
    box[:, :4] = clip_boxes_to_image(box[:, :4], meta.ori_size)

    return box


def area(box):
    return (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])

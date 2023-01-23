import torch
from torch import nn
from torch.nn import functional as F


def focal_loss(x, target, mask=None, gamma=1.0, reduction="sum"):
    if x.shape[0] == 0:
        return 0 * x.sum()
    # 1. One-hot target
    if target.dim() == 1:
        target = F.one_hot(target, num_classes=x.shape[1]).float()

    # 2. Generate Mask
    if mask is None:
        mask = torch.ge(target, 0).float()
    else:
        assert mask.shape == target.shape, "Mask shape must be equal to target shape!"
        mask = mask.float()

    exp_x = x.exp()
    pos_term = (1 / exp_x * target * mask).sum(dim=1)
    neg_term = (exp_x * torch.eq(target, 0).float() * mask).sum(dim=1)

    CE = (1 + pos_term * neg_term).log()

    p = torch.exp(-CE)
    loss = (1 - p) ** gamma * CE

    if reduction == "sum":
        return loss.sum()
    elif reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise ValueError("Unsupported reduction")

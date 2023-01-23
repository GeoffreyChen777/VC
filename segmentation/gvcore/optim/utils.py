from typing import Dict, List
import torch


def sum_loss_list(losses_list):
    summed_loss = torch.zeros((1,), device="cuda")
    for loss in losses_list:
        if loss is not None:
            summed_loss += loss
    return summed_loss


def sum_loss_dict(losses_dict):
    return sum_loss_list(list(losses_dict.values()))


def sum_loss(losses, weight=None):
    if weight is not None:
        losses = weight_loss(losses, weight)

    if isinstance(losses, dict):
        return sum_loss_dict(losses)
    elif isinstance(losses, list):
        return sum_loss_list(losses)
    else:
        raise TypeError


def merge_loss_dict(losses_dict_a, losses_dict_b, mean=False):
    losses_dict = {}
    a = 0.5 if mean else 1
    for name in losses_dict_a.keys():
        losses_dict[name] = a * (losses_dict_a[name] + losses_dict_b[name])
    return losses_dict


def weight_loss(losses: Dict, weights: Dict):
    assert isinstance(losses, dict) == isinstance(weights, dict), "losses and weights must be both dict or both list"
    assert isinstance(losses, list) == isinstance(losses, list), "losses and weights must be both dict or both list"

    if isinstance(losses, dict):
        for name in losses.keys():
            losses[name] *= weights.get(name, 1)
    else:
        assert len(losses) == len(weights), "losses and weights must be same length"
        for i, (loss, weight) in enumerate(zip(losses, weights)):
            losses[i] = loss * weight
    return losses

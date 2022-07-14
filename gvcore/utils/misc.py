import torch
import time


def time_test(func):
    def wrapper(*args):
        st = time.time()
        out = func(*args)
        print(args[0].__class__.__name__, "{:.8f}".format(time.time() - st))
        return out

    return wrapper


def sum_loss_list(losses_list, weight):
    summed_loss = torch.zeros((1,), device="cuda")
    if weight is not None:
        for loss, w in zip(losses_list, weight):
            if loss is not None:
                summed_loss += loss * w
    else:
        for loss in losses_list:
            if loss is not None:
                summed_loss += loss
    return summed_loss


def sum_loss_dict(losses_dict, weight):
    return sum_loss_list(list(losses_dict.values()), weight)


def sum_loss(losses, weight=None):
    if isinstance(losses, dict):
        return sum_loss_dict(losses, weight)
    elif isinstance(losses, list):
        return sum_loss_list(losses, weight)
    else:
        raise TypeError


def merge_loss_dict(losses_dict_a, losses_dict_b, mean=False):
    losses_dict = {}
    a = 0.5 if mean else 1
    for name in losses_dict_a.keys():
        losses_dict[name] = a * (losses_dict_a[name] + losses_dict_b[name])
    return losses_dict


def weight_loss(losses, weights):
    for loss_name, loss_value in losses.items():
        if loss_name in weights and loss_value is not None:
            losses[loss_name] *= weights[loss_name]
    return losses


def attach_batch_idx(tensor_list):
    tensors = []
    for i, tensor in enumerate(tensor_list):
        batch_idx = torch.ones((tensor.shape[0], 1), dtype=tensor.dtype, device=tensor.device) * i
        tensors.append(torch.cat((batch_idx, tensor), dim=1))
    return torch.cat(tensors, dim=0)


def print_tensor(x, device="cuda:0"):
    if isinstance(x, torch.Tensor):
        if str(x.device) == device:
            print(x)
    else:
        print(x)


def slerp(low, high, val=0.5):
    omega = torch.arccos((low * high).sum(dim=1, keepdim=True))
    so = torch.sin(omega)
    return torch.sin((1.0 - val) * omega) / so * low + torch.sin(val * omega) / so * high


def entropy(x):
    p = torch.softmax(x, dim=1)
    entropy = -(p * p.log()).sum(dim=1)
    return entropy


def split_ab(x):
    if len(x) <= 1:
        return x, x
    a = x[: int(len(x) / 2)]
    b = x[int(len(x) / 2) :]

    return a, b

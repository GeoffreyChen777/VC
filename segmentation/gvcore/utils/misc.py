from typing import List, Sequence
import torch
import time

from gvcore.utils.structure import GenericData


def time_test(func):
    def wrapper(*args):
        st = time.time()
        out = func(*args)
        print(args[0].__class__.__name__, "{:.8f}".format(time.time() - st))
        return out

    return wrapper


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


def sharpen(x: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    sharpened_x = x ** (1 / temperature)
    return sharpened_x / sharpened_x.sum(dim=1, keepdim=True)


def interleave_offsets(batch_size: int, num_unlabeled: int) -> List[int]:
    # TODO: scrutiny
    groups = [batch_size // (num_unlabeled + 1)] * (num_unlabeled + 1)
    for x in range(batch_size - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch_size
    return offsets


def interleave(xy: Sequence[GenericData], batch_size: int) -> List[GenericData]:
    # TODO: scrutiny
    num_unlabeled = len(xy) - 1
    offsets = interleave_offsets(batch_size, num_unlabeled)
    xy = [[v[offsets[p] : offsets[p + 1]] for p in range(num_unlabeled + 1)] for v in xy]
    for i in range(1, num_unlabeled + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]

    outs = []
    for v in xy:
        if torch.is_tensor(v[0]):
            v = torch.cat(v, dim=0)
        else:
            v = [item for subv in v for item in subv]
        outs.append(v)
    return outs


def one_hot(y, num_classes):
    y_tensor = y.unsqueeze(1)
    zeros = torch.zeros([y.shape[0], num_classes] + list(y.shape[1:]), dtype=y.dtype, device=y.device)

    return zeros.scatter(1, y_tensor, 1)

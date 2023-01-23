from collections import deque
from addict import Dict
import random
import torch
import copy

from gvcore.utils.distributed import all_gather


class GenericData:
    img: torch.Tensor
    label: torch.Tensor
    meta: Dict

    def has(self, key):
        return hasattr(self, key)

    def set(self, key, value):
        setattr(self, key, value)

    def get(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return None

    def remove(self, key):
        if hasattr(self, key):
            delattr(self, key)

    def to(self, device):
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                self.set(key, value.to(device, non_blocking=True))
            elif isinstance(value, GenericData):
                value.to(device)
        return self

    def clone(self):
        return copy.deepcopy(self)

    def _repr(self, parent=""):
        repr = ""
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor):
                value_repr = f"{parent}{key}: {value.dtype}, {list(value.size())}\n"
            elif isinstance(value, GenericData):
                value_repr = value._repr(key + "." + parent)
            else:
                value_repr = f"{parent}{key}: {value}\n"
            repr += value_repr
        return repr

    def __repr__(self):
        return "\n(\n" + self._repr() + ")\n"


class TensorQueue:
    def __init__(self, size):
        self.size = size

        self.queue = deque(maxlen=size)

    def push(self, x):
        self.queue.append(x.to("cpu"))

    def sample(self, n: int):
        n = min(n, len(self.queue))
        return random.sample(self.queue, n)

    def sum(self):
        return torch.cat(list(self.queue)).sum()

    def mean(self):
        return torch.cat(list(self.queue)).mean()


class TensorDict:
    def __init__(self):
        self.dict = {}

    def __missing__(self, key):
        return self[str(key)]

    def __contains__(self, key):
        return str(key) in self.dict

    def __setitem__(self, key, item):
        ks = all_gather(key)
        vs = all_gather(item)

        for k, v in zip(ks, vs):
            self.dict[str(k)] = v.cpu()

    def __getitem__(self, key):
        return self.dict[str(key)]

    def __str__(self) -> str:
        return str(self.dict)


class TensorList:
    def __init__(self, size, init_value=-1, device="cpu", dtype=torch.long, all_gather=True):
        self._tensor = torch.zeros(size, device=device, dtype=dtype).fill_(init_value)
        self.all_gather = all_gather

    def __setitem__(self, idx, item):
        if self.all_gather:
            ks = all_gather(idx)
            vs = all_gather(item)

            ks = torch.cat(ks)
            vs = torch.cat(vs)
        else:
            ks = idx
            vs = item

        self._tensor.index_copy_(0, ks.to(self._tensor.device), vs.to(dtype=self._tensor.dtype, device=self._tensor.device))

    def __getitem__(self, idx):
        return self.matrix[idx.to(self.matrix.device)]


class TensorMatrix:
    def __init__(self, size, init_value=-1, device="cpu", dtype=torch.long):
        self.matrix = torch.zeros(size, device=device, dtype=dtype).fill_(init_value)

    def __setitem__(self, idx, item):
        ks = all_gather(idx)
        vs = all_gather(item)

        ks = torch.cat(ks)
        vs = torch.cat(vs)

        self.matrix.scatter_(0, ks.to(self.matrix.device), vs.to(self.matrix.device))

    def __getitem__(self, idx):
        return self.matrix[idx.to(self.matrix.device)]


class ConfusionMatrix:
    def __init__(self, num_classes, iter_n, device="cpu") -> None:
        self.m = torch.zeros((iter_n, num_classes, num_classes), device=device)
        self.i = 0

        self.num_classes = num_classes

    @torch.no_grad()
    def push(self, cls_a, cls_b) -> None:
        cls_a_list = all_gather(cls_a)
        cls_b_list = all_gather(cls_b)
        cls_a = torch.cat(cls_a_list)
        cls_b = torch.cat(cls_b_list)
        i = self.i % self.m.shape[0]

        m = torch.zeros((self.num_classes, self.num_classes), device=cls_a.device)
        m[cls_a.long(), cls_b.long()] += 1

        self.m[i] = m

        self.i += 1

    @torch.no_grad()
    def summary(self):
        return self.m.sum(dim=0)

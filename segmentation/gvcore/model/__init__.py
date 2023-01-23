import torch
import torch.nn as nn

from gvcore.utils.registry import Registry

MODEL_REGISTRY = Registry()

class GenericModule(nn.Module):
    def __init__(self, cfg=None):
        super(GenericModule, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)

    def forward_generic(self, *args, **kwargs):
        raise NotImplementedError

    def forward_train(self, *args, **kwargs):
        return self.forward_generic(*args, **kwargs)

    def forward_eval(self, *args, **kwargs):
        return self.forward_generic(*args, **kwargs)

    def get_losses(self, *args, **kwargs):
        raise NotImplementedError

    def freeze(self, if_freeze=True):
        for p in self.parameters():
            p.requires_grad = not if_freeze

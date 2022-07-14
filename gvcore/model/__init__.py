import torch
import torch.nn as nn


class GenericModule(nn.Module):
    def __init__(self, cfg):
        super(GenericModule, self).__init__()
        self.cfg = cfg
        self.device = torch.device("cuda")

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)

    def forward_train(self, *args, **kwargs):
        raise NotImplementedError

    def forward_eval(self, *args, **kwargs):
        raise NotImplementedError

    def get_losses(self, *args, **kwargs):
        raise NotImplementedError

    def freeze(self, if_freeze=True):
        for p in self.parameters():
            p.requires_grad = not if_freeze

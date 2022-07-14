import torch
import math
import torch.optim.lr_scheduler as lr_s
from bisect import bisect_right


class PolyPolicy:
    def __init__(self, iter_num, power=0.9):
        self.power = power
        self.iter_num = iter_num

    def __call__(self, step):
        return (1 - step / self.iter_num) ** self.power


class PolyLR(lr_s.LambdaLR):
    def __init__(self, optimizer, iter_num, power=0.9):
        self.power = power
        self.iter_num = iter_num
        poly = PolyPolicy(self.iter_num, self.power)
        super(PolyLR, self).__init__(optimizer, poly)


class WarmupPolicy:
    def __init__(self, warmup_step, warmup_gamma):
        self.warmup_step = warmup_step
        self.warmup_gamma = warmup_gamma

    def __call__(self, step):
        if step >= self.warmup_step:
            return 1
        alpha = float(step) / self.warmup_step
        return self.warmup_gamma * (1 - alpha) + alpha


class WarmupLR(lr_s.LambdaLR):
    def __init__(self, optimizer, warmup_step, warmup_gamma):
        policy = WarmupPolicy(warmup_step, warmup_gamma)
        super(WarmupLR, self).__init__(optimizer, policy)


class MultiStepPolicy:
    def __init__(self, milestones, gamma, warmup=False, warmup_step=None, warmup_gamma=None):
        self.milestones = milestones
        self.gamma = gamma
        self.warmup = warmup
        if self.warmup:
            assert warmup_step is not None and warmup_gamma is not None, "Warmup policy need to set step and gamma!"
            self.warmup_policy = WarmupPolicy(warmup_step, warmup_gamma)
            self.warmup_step = warmup_step
        else:
            self.warmup_step = -1

    def __call__(self, step):
        if step <= self.warmup_step:
            return self.warmup_policy(step)
        else:
            return self.gamma ** bisect_right(self.milestones, step)


class MultiStepLR(lr_s.LambdaLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup=False, warmup_step=None, warmup_gamma=None):
        policy = MultiStepPolicy(milestones, gamma, warmup, warmup_step, warmup_gamma)
        super(MultiStepLR, self).__init__(optimizer, policy)


class CosinePolicy:
    def __init__(self, iter_num, num_cycles=7./16., warmup=False, warmup_step=None, warmup_gamma=None):
        self.iter_num = iter_num
        self.num_cycles = num_cycles
        self.warmup = warmup
        if self.warmup:
            assert warmup_step is not None and warmup_gamma is not None, "Warmup policy need to set step and gamma!"
            self.warmup_policy = WarmupPolicy(warmup_step, warmup_gamma)
            self.warmup_step = warmup_step
        else:
            self.warmup_step = 0

    def __call__(self, step):
        if step < self.warmup_step:
            return self.warmup_policy(step)
        else:
            return math.cos(math.pi * self.num_cycles * (step - self.warmup_step) / (self.iter_num - self.warmup_step))


class CosineLR(lr_s.LambdaLR):
    def __init__(self, optimizer, iter_num, num_cycles=7./16., warmup=False, warmup_step=None, warmup_gamma=None):
        policy = CosinePolicy(iter_num, num_cycles, warmup, warmup_step, warmup_gamma)
        super(CosineLR, self).__init__(optimizer, policy)

import torch.nn as nn
import torch


def xavier_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # Caffe2 implementation of XavierFill in fact
    # corresponds to kaiming_uniform_ in PyTorch
    nn.init.kaiming_uniform_(module.weight, a=1)  # pyre-ignore
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def msra_fill(module: nn.Module) -> None:
    """
    Initialize `module.weight` using the "MSRAFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.

    Args:
        module (torch.nn.Module): module to initialize.
    """
    # pyre-ignore
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


def kaiming_init(
    module: nn.Module,
    a: float = 0,
    mode: str = "fan_out",
    nonlinearity: str = "relu",
    bias: float = 0,
    distribution: str = "normal",
) -> None:
    assert distribution in ["uniform", "normal"]
    if hasattr(module, "weight") and module.weight is not None:
        if distribution == "uniform":
            nn.init.kaiming_uniform_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
        else:
            nn.init.kaiming_normal_(module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def get_default_optimizer_params(
    model: torch.nn.Module,
    base_lr,
    weight_decay,
    weight_decay_norm,
    bias_lr_factor=1.0,
    weight_decay_bias=None,
    overrides=None,
):
    """
    Get default param list for optimizer

    """
    if weight_decay_bias is None:
        weight_decay_bias = weight_decay
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        # NaiveSyncBatchNorm inherits from BatchNorm2d
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    params = []
    memo = set()
    for module in model.modules():
        for module_param_name, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)

            schedule_params = {"lr": base_lr, "weight_decay": weight_decay}
            if isinstance(module, norm_module_types):
                schedule_params["weight_decay"] = weight_decay_norm
            elif module_param_name == "bias":
                # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                # hyperparameters are by default exactly the same as for regular
                # weights.
                schedule_params["lr"] = base_lr * bias_lr_factor
                schedule_params["weight_decay"] = weight_decay_bias
            if overrides is not None and module_param_name in overrides:
                schedule_params.update(overrides[module_param_name])
            params += [
                {"params": [value], "lr": schedule_params["lr"], "weight_decay": schedule_params["weight_decay"]}
            ]

    return params


@torch.no_grad()
def match_label(boxes, labels, matcher):
    matches = []
    match_flags = []
    if isinstance(boxes, list):
        for b, label in zip(boxes, labels):
            label_box = label[:, :4]
            match, match_flag = matcher(b, label_box)
            matches.append(match)
            match_flags.append(match_flag)
    else:
        for label in labels:
            label_box = label[:, :4]
            match, match_flag = matcher(boxes, label_box)
            matches.append(match)
            match_flags.append(match_flag)
    return matches, match_flags


@torch.no_grad()
def sample_pos_neg(match_flags, sample_num, pos_fraction):
    for i, match_flag in enumerate(match_flags):
        positive = torch.nonzero(match_flag > 0, as_tuple=True)[0]
        negative = torch.nonzero(match_flag == 0, as_tuple=True)[0]

        num_pos = int(sample_num * pos_fraction)
        # protect against not enough positive examples
        num_pos = min(positive.numel(), num_pos)
        num_neg = sample_num - num_pos
        # protect against not enough negative examples
        num_neg = min(negative.numel(), num_neg)

        # randomly select positive and negative examples
        perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
        perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

        pos_idx = positive[perm1]
        neg_idx = negative[perm2]

        match_flag.fill_(-1)
        match_flag.scatter_(0, pos_idx, 1)
        match_flag.scatter_(0, neg_idx, 0)

        match_flags[i] = match_flag
    return match_flags


def freeze(model, if_freeze):
    model.requires_grad_(not if_freeze)

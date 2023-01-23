import torch.nn as nn

import gvcore.utils.distributed as dist_utils


def Normalize(type: str):
    if type == "BN":
        return nn.BatchNorm2d
    elif type == "GN":
        return nn.GroupNorm
    elif type == "LN":
        return nn.LayerNorm
    elif type == "SyncBN":
        if dist_utils.get_world_size() > 1:
            return nn.SyncBatchNorm
        else:
            return nn.BatchNorm2d
    elif type == "none":
        return None
    else:
        raise ValueError("Unknown norm type {}".format(type))

import torch
import torch.nn as nn
import math
from torchvision.ops import sigmoid_focal_loss
from model.anchor import BoxCoder
import torch.nn.functional as F


class RetinaNetHead(nn.Module):
    """
    A regression and classification head for use in RetinaNet.
    Arguments:
        in_channels (int): number of channels of the input feature
        num_anchors (int): number of anchors to be predicted
        num_classes (int): number of classes to be predicted
    """

    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()

        cls_subnet = []
        bbox_subnet = []
        for _ in range(4):
            cls_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_subnet.append(nn.ReLU())
            bbox_subnet.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            bbox_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.bbox_subnet = nn.Sequential(*bbox_subnet)
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)

        self._init_modules()

        self.num_classes = num_classes
        self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        self.loss_normalizer_cls = 100  # For semi training
        self.loss_normalizer_reg = 100  # For semi training
        self.loss_normalizer_momentum = 0.9

        self.box_coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))

    def _init_modules(self, prior_probability=0.01):
        for modules in [self.cls_subnet, self.bbox_subnet, self.cls_score, self.bbox_pred]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - prior_probability) / prior_probability))
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, features):
        logits = []
        bbox_reg = []
        for feature in features.values():
            logit = self.cls_score(self.cls_subnet(feature))
            n, _, h, w = logit.shape
            logit = logit.view(n, -1, self.num_classes, h, w)
            logit = logit.permute(0, 3, 4, 1, 2)
            logit = logit.reshape(n, -1, self.num_classes)  # Size=(N, HWA, 4)
            logits.append(logit)

            reg = self.bbox_pred(self.bbox_subnet(feature))
            # Permute bbox regression output from (N, 4 * A, H, W) to (N, HWA, 4).
            n, _, h, w = reg.shape
            reg = reg.view(n, -1, 4, h, w)
            reg = reg.permute(0, 3, 4, 1, 2)
            reg = reg.reshape(n, -1, 4)  # Size=(N, HWA, 4)
            bbox_reg.append(reg)

        return {"cls_logits": logits, "bbox_regression": bbox_reg}

    @staticmethod
    def smooth_l1_loss(input: torch.Tensor, target: torch.Tensor, beta: float, reduction: str = "none") -> torch.Tensor:
        """
        Smooth L1 loss defined in the Fast R-CNN paper as:

                      | 0.5 * x ** 2 / beta   if abs(x) < beta
        smoothl1(x) = |
                      | abs(x) - 0.5 * beta   otherwise,

        where x = input - target.

        Smooth L1 loss is related to Huber loss, which is defined as:

                    | 0.5 * x ** 2                  if abs(x) < beta
         huber(x) = |
                    | beta * (abs(x) - 0.5 * beta)  otherwise

        Smooth L1 loss is equal to huber(x) / beta. This leads to the following
        differences:

         - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
           converges to a constant 0 loss.
         - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
           converges to L2 loss.
         - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
           slope of 1. For Huber loss, the slope of the L1 segment is beta.

        Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
        portion replaced with a quadratic function such that at abs(x) = beta, its
        slope is 1. The quadratic segment smooths the L1 loss near x = 0.

        Args:
            input (Tensor): input tensor of any shape
            target (Tensor): target value tensor with the same shape as input
            beta (float): L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            reduction: 'none' | 'mean' | 'sum'
                     'none': No reduction will be applied to the output.
                     'mean': The output will be averaged.
                     'sum': The output will be summed.

        Returns:
            The loss with the reduction option applied.

        Note:
            PyTorch's builtin "Smooth L1 loss" implementation does not actually
            implement Smooth L1 loss, nor does it implement Huber loss. It implements
            the special case of both in which they are equal (beta=1).
            See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
        """
        if beta < 1e-5:
            # if beta == 0, then torch.where will result in nan gradients when
            # the chain rule is applied due to pytorch implementation details
            # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
            # zeros, rather than "no gradient"). To avoid this issue, we define
            # small values of beta to be exactly l1 loss.
            loss = torch.abs(input - target)
        else:
            n = torch.abs(input - target)
            cond = n < beta
            loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        return loss

    def cal_loss(self, preds, labels, anchors, matches, match_flags):
        cls_logits = preds["cls_logits"]
        box_regs = preds["bbox_regression"]

        cls_labels, reg_labels, paired_anchors = [], [], []
        fg_flags, valid_flags = [], []
        for label, match, match_flag in zip(labels, matches, match_flags):
            fg_flag = match_flag > 0
            bg_flag = match_flag == 0
            valid_flag = match_flag >= 0

            fg_flags.append(fg_flag)
            valid_flags.append(valid_flag)

            if label.shape[0] == 0:
                continue

            match_label = label[match]

            cls_label = match_label[:, -1]
            cls_label[bg_flag] = self.num_classes
            cls_label = cls_label[valid_flag]

            box_label = match_label[fg_flag, :4]
            reg_label = self.box_coder.encode_single(box_label, anchors[fg_flag])

            cls_labels.append(cls_label)
            reg_labels.append(reg_label)

            del bg_flag

        cls_labels = torch.cat(cls_labels)
        reg_labels = torch.cat(reg_labels)
        fg_flags = torch.stack(fg_flags)
        # bg_flags = torch.stack(bg_flags)
        valid_flags = torch.stack(valid_flags)

        # Calculate momentum
        num_fg = fg_flags.sum().item()
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
            1 - self.loss_normalizer_momentum
        ) * max(num_fg, 1)

        # ===========================================
        # Classification
        # Filter valid logits.
        valid_cls_logits = cls_logits[valid_flags]

        # Create classification one_hot target.
        cls_labels = F.one_hot(cls_labels.long(), num_classes=self.num_classes + 1)[:, :-1].to(valid_cls_logits.dtype)

        # compute the classification loss
        cls_loss = sigmoid_focal_loss(valid_cls_logits, cls_labels, alpha=0.25, gamma=2.0, reduction="sum")
        # print(cls_loss, self.loss_normalizer, num_fg, valid_flags.sum())

        # ===========================================
        # Regression
        # Filter valid regression.
        valid_box_regs = box_regs[fg_flags]
        # compute the regression loss
        reg_loss = self.smooth_l1_loss(valid_box_regs, reg_labels, beta=0.0, reduction="sum")
        return {"cls": cls_loss / self.loss_normalizer, "reg": reg_loss / self.loss_normalizer}

    def cal_loss_semi(self, preds, labels, anchors, matches, match_flags):
        cls_logits = preds["cls_logits"]
        box_regs = preds["bbox_regression"]

        cls_labels, reg_labels, paired_anchors = [], [], []
        cls_valid_fg_flags, cls_valid_flags, loc_valid_flags = [], [], []
        for label, match, match_flag in zip(labels, matches, match_flags):
            fg_flag = match_flag > 0
            bg_flag = match_flag == 0
            cls_valid_flag = match_flag >= 0

            match_label = label[match]
            defined_cls_valid_flag = match_label[:, -2].bool()
            defined_loc_valid_flag = match_label[:, -1].bool()
            cls_valid_flag *= defined_cls_valid_flag
            loc_valid_flag = defined_loc_valid_flag * fg_flag
            cls_valid_fg_flag = fg_flag * cls_valid_flag

            cls_label = match_label[:, -3]
            cls_label[bg_flag] = self.num_classes
            cls_label = cls_label[cls_valid_flag]

            box_label = match_label[loc_valid_flag, :4]
            reg_label = self.box_coder.encode_single(box_label, anchors[loc_valid_flag])

            cls_labels.append(cls_label)
            reg_labels.append(reg_label)

            cls_valid_fg_flags.append(cls_valid_fg_flag)
            # bg_flags.append(bg_flag)
            cls_valid_flags.append(cls_valid_flag)
            loc_valid_flags.append(loc_valid_flag)
            del bg_flag

        cls_labels = torch.cat(cls_labels)
        reg_labels = torch.cat(reg_labels)
        cls_valid_fg_flags = torch.stack(cls_valid_fg_flags)
        cls_valid_flags = torch.stack(cls_valid_flags)
        loc_valid_flags = torch.stack(loc_valid_flags)

        # Calculate momentum
        num_fg_cls = cls_valid_fg_flags.sum().item()
        self.loss_normalizer_cls = self.loss_normalizer_momentum * self.loss_normalizer_cls + (
            1 - self.loss_normalizer_momentum
        ) * max(num_fg_cls, 1)
        num_reg = loc_valid_flags.sum().item()
        self.loss_normalizer_reg = self.loss_normalizer_momentum * self.loss_normalizer_reg + (
            1 - self.loss_normalizer_momentum
        ) * max(num_reg, 1)
        # self.loss_normalizer = max(num_fg, 1)

        # ===========================================
        # Classification
        # Filter valid logits.
        valid_cls_logits = cls_logits[cls_valid_flags]

        # Create classification one_hot target.
        cls_labels = F.one_hot(cls_labels.long(), num_classes=self.num_classes + 1)[:, :-1].to(valid_cls_logits.dtype)

        # compute the classification loss
        cls_loss = sigmoid_focal_loss(valid_cls_logits, cls_labels, alpha=0.25, gamma=2.0, reduction="sum")

        # ===========================================
        # Regression
        # Filter valid regression.
        valid_box_regs = box_regs[loc_valid_flags]
        # compute the regression loss
        reg_loss = self.smooth_l1_loss(valid_box_regs, reg_labels, beta=0.0, reduction="sum")
        return {"cls": cls_loss / self.loss_normalizer_cls, "reg": reg_loss / self.loss_normalizer_reg}

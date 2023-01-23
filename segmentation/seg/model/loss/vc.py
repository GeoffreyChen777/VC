from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from gvcore.utils.misc import entropy


class VCLoss:
    def __init__(
        self,
        confidence_threshold: float = 0.0,
        type: str = "mse",
        num_classes: Optional[int] = None,
        gamma: float = 0.0,
    ):
        if type == "mse" or type == "mean_squared_error":
            self.loss_fn = self._mse
        elif type == "ce" or type == "cross_entropy":
            self.loss_fn = self._cross_entropy
        elif type == "soft_ce" or type == "soft_cross_entropy":
            self.loss_fn = self._soft_cross_entropy
        elif type == "focal_loss":
            self.loss_fn = self._focal_loss
            self.gamma = gamma
        elif type == "mse_full_vc":
            self.loss_fn = self._mse_full_vc
        else:
            raise ValueError("Type {} is not supported.".format(type))

        self.confidence_threshold = confidence_threshold
        self.num_classes = num_classes

    def _avoid_nan_backward_hook(self, logits: torch.Tensor):
        logits.register_hook(lambda grad: torch.nan_to_num(grad, 0, 0, 0))

    def _masked_softmax(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = torch.ge(targets, 0)

        logits = logits - logits.max(dim=1, keepdim=True)[0]
        logits_exp = torch.exp(logits)
        probs = logits_exp / ((logits_exp * mask.float()).sum(dim=1, keepdim=True) + 1e-10)

        return probs, mask

    def _mse(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        probs, mask = self._masked_softmax(logits, targets)

        mse = F.mse_loss(probs, targets, reduction="none") * mask.float()

        if redunction == "mean":
            mse = (mse.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            mse = mse.sum(dim=1)
        elif redunction == "none":
            pass

        return mse

    def _mse_full_vc(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        non_vc_flag = (targets[:, -1] == -1).unsqueeze(1)

        targets_max, targets_max_idx = targets.max(dim=1, keepdim=True)
        vc_pos_idx = torch.cat((targets_max_idx, targets_max_idx), dim=1)
        vc_pos_idx[:, -1] = targets.shape[1] - 1

        vc_pos_idx[~non_vc_flag.expand_as(vc_pos_idx)] = targets.shape[1] - 1
        targets_max[non_vc_flag.expand_as(targets_max)] = targets_max[non_vc_flag.expand_as(targets_max)] / 2.0

        targets = targets.scatter(1, vc_pos_idx, targets_max.expand_as(vc_pos_idx))

        probs, mask = self._masked_softmax(logits, targets)

        mse = F.mse_loss(probs, targets, reduction="none") * mask.float()

        if redunction == "mean":
            mse = (mse.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            mse = mse.sum(dim=1)
        elif redunction == "none":
            pass

        return mse

    def _focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[0] == 0:
            return 0 * logits.sum()

        mask = torch.ge(targets, 0).float()

        logits_exp = logits.exp()

        pos_term = (1 / logits_exp * targets * mask).sum(dim=1)
        neg_term = (logits_exp * torch.eq(targets, 0).float() * mask).sum(dim=1)

        ce = (1 + pos_term * neg_term).log()

        p = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce

        return loss

    def _cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        mask = torch.ge(targets, 0).float()

        logits_exp = logits.exp()

        pos_term = (1 / logits_exp * targets * mask).sum(dim=1)
        neg_term = (logits_exp * torch.eq(targets, 0).float() * mask).sum(dim=1)

        ce = (1 + pos_term * neg_term).log()

        if redunction == "mean":
            ce = (ce.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            ce = ce.sum(dim=1)
        elif redunction == "none":
            pass

        return ce

    def _soft_cross_entropy(self, logits: torch.Tensor, targets: torch.Tensor, redunction="none") -> torch.Tensor:
        mask = torch.ge(targets, 0).float()

        logits_exp = (logits - (logits.max(dim=1, keepdim=True)[0] - logits.min(dim=1, keepdim=True)[0]) / 2.0).exp()
        probs = logits_exp / (logits_exp * mask.float()).sum(dim=1, keepdim=True)

        probs_log = probs.log()

        ce = (-targets * probs_log) * mask.float()

        if redunction == "mean":
            ce = (ce.sum(dim=1) / mask.sum(dim=1).float()).mean()
        elif redunction == "sum":
            ce = ce.sum(dim=1)
        elif redunction == "none":
            pass

        return ce

    def _get_target_clses_and_probs(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() - 1 == targets.dim():
            clses = targets
            num_classes = self.num_classes if self.num_classes is not None else logits.shape[1] - 1
            probs = torch.zeros((logits.shape[0], num_classes, *logits.shape[2:]), device=logits.device).scatter_(
                1, targets.unsqueeze(1), 1
            )
        else:
            clses = targets.argmax(dim=1)
            probs = targets

        return clses, probs

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor, potential_targets: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C+1, *)
            targets: (B, *) or (B, C, *), i.e., clses or probs
            potential_targets: (B, *) or (B, C, *)
            reduction: 'none', 'mean', 'sum'
        
        Returns:
            loss: (B, *)

        """
        assert logits.dim() == targets.dim() or logits.dim() - 1 == targets.dim(), "logits or targets shape is invalid."
        assert (
            logits.dim() == potential_targets.dim() or logits.dim() - 1 == potential_targets.dim()
        ), "logits or potential_targets shape is invalid."
        self._avoid_nan_backward_hook(logits)

        with torch.no_grad():
            t_clses, t_probs = self._get_target_clses_and_probs(logits, targets)
            pt_clses, pt_probs = self._get_target_clses_and_probs(logits, potential_targets)

            vc_target = t_probs.clone()

            ignore_clses = torch.stack((t_clses, pt_clses), dim=1)

            vc_prob = torch.gather(t_probs, 1, ignore_clses).sum(dim=1, keepdim=True)
            vc_target = torch.cat((vc_target, vc_prob), dim=1)

            no_vc_mask = torch.all(torch.eq(ignore_clses, ignore_clses[:, 0].unsqueeze(1)), dim=1)
            ignore_clses.masked_fill_(no_vc_mask.unsqueeze(1), logits.shape[1] - 1)

            vc_target = vc_target.scatter(1, ignore_clses, -1)

        loss = self.loss_fn(logits, vc_target)

        if self.confidence_threshold > 0.0:
            confidence_mask = torch.ge(t_probs.max(dim=1)[0], self.confidence_threshold)
            loss = loss * confidence_mask.unsqueeze(1).float()

        loss = torch.nan_to_num(loss, 0, 0, 0)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum(dim=1)
        elif reduction == "none":
            pass

        return loss


class ProbVCLoss(VCLoss):
    def __init__(
        self,
        confidence_threshold: float = 0.0,
        type: str = "mse",
        num_classes: Optional[int] = None,
        ignore_topk: int = 2,
    ):
        assert (
            type != "ce" and type != "cross_entropy"
        ), "ProbVCLoss does not support cross entropy, use soft cross entropy instead."
        super().__init__(confidence_threshold, type, num_classes)

        self.ignore_topk = ignore_topk

    def _get_potential_clses_and_probs(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert targets.shape[1] > self.ignore_topk, "targets shape is invalid."

        probs, clses = torch.sort(targets, dim=1, descending=True)
        clses = clses[:, : self.ignore_topk]
        probs = probs[:, : self.ignore_topk]

        return clses, probs

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor, threshold: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C+1, *)
            targets: (B, C, *), i.e., probs
            threshold: (B,)
            reduction: 'none', 'mean', 'sum'
        
        Returns:
            loss: (B, *)

        """

        assert logits.dim() == targets.dim(), "logits or targets shape is invalid."

        with torch.no_grad():
            ignore_clses, ignore_probs = self._get_potential_clses_and_probs(targets)
            top1_probs = ignore_probs[:, 0]

            vc_mask = torch.le(top1_probs, threshold)

            vc_target = targets.clone()
            vc_target = torch.cat((vc_target, ignore_probs.sum(dim=1, keepdim=True)), dim=1)

            ignore_clses.masked_fill_(vc_mask.logical_not().unsqueeze(1), logits.shape[1] - 1)

            vc_target = vc_target.scatter(1, ignore_clses, -1)

        loss = self.loss_fn(logits, vc_target, redunction=reduction)

        return loss


class EntropyVCLoss(VCLoss):
    def __init__(self, ignore_topk: int = 3, type: str = "mse"):
        super().__init__(type=type)

        self.ignore_topk = ignore_topk

    def _get_potential_clses_and_probs(self, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert targets.shape[1] > self.ignore_topk, "targets shape is invalid."

        probs, clses = torch.sort(targets, dim=1, descending=True)
        clses = clses[:, : self.ignore_topk]
        probs = probs[:, : self.ignore_topk]

        return clses, probs

    def __call__(
        self, logits: torch.Tensor, targets: torch.Tensor, threshold: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        """
        Args:
            logits: (B, C+1, *)
            targets: (B, C, *), i.e., probs
            threshold: (B,)
            reduction: 'none', 'mean', 'sum'
        
        Returns:
            loss: (B, *)

        """

        assert logits.dim() == targets.dim(), "logits or targets shape is invalid."

        ignore_clses, ignore_probs = self._get_potential_clses_and_probs(targets)

        with torch.no_grad():
            targets_entropy = entropy(targets)
            if threshold.dim() == 1:
                threshold = threshold.unsqueeze(1).unsqueeze(2)
            vc_mask = torch.gt(targets_entropy, threshold)

        vc_target = targets.clone()
        vc_target = torch.cat((vc_target, ignore_probs.sum(dim=1, keepdim=True)), dim=1)

        ignore_clses.masked_fill_(vc_mask.logical_not().unsqueeze(1), logits.shape[1] - 1)

        vc_target = vc_target.scatter(1, ignore_clses, -1)

        loss = self.loss_fn(logits, vc_target, redunction=reduction)

        return loss

from copy import deepcopy
from typing import Callable, Optional
from addict import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


from gvcore.utils.types import TDataList
from gvcore.utils.misc import sharpen
from gvcore.model import GenericModule

from model.loss.vc import VCLoss
from model.segmentor.ts import TSSegmentor


class WeightGenerator(GenericModule):
    def __init__(self, feats_channels=256) -> None:
        super().__init__()

        self.feats_channels = feats_channels

        self.transformer_encoder = nn.TransformerEncoderLayer(
            d_model=feats_channels, nhead=1, dim_feedforward=64, batch_first=True
        )

    def forward_train(
        self,
        init_feats: torch.Tensor,
        init_weights: torch.Tensor,
        labels: torch.Tensor = None,
        scores: torch.Tensor = None,
    ) -> torch.Tensor:

        if labels is not None:
            labels = (
                F.interpolate(labels.unsqueeze(1).float(), size=init_feats.shape[2:], mode="nearest").squeeze(1).long()
            )
            scores = F.interpolate(
                scores.unsqueeze(1), size=init_feats.shape[2:], mode="bilinear", align_corners=False
            ).squeeze(1)

            sampled_mask = torch.zeros_like(labels, dtype=torch.bool)

            # uniform sample labels by classes.
            unique_classes, unique_n = labels.unique(return_counts=True)
            # each class samples top 20% according to scores
            for cls, n in zip(unique_classes, unique_n):
                if cls.item() == 255:
                    continue
                cls_scores = scores[labels == cls]
                cls_scores = cls_scores.sort(descending=False)[0]
                cls_threshold = cls_scores[int(n * 0.2)]
                sampled_mask = torch.where(labels == cls, scores < cls_threshold, sampled_mask)

            feats = init_feats.permute(0, 2, 3, 1).reshape(-1, 1, self.feats_channels)[sampled_mask.view(-1)]
            init_weights = init_weights.view(1, -1, self.feats_channels)

            x = torch.cat((feats, init_weights.repeat(feats.shape[0], 1, 1)), dim=1)

            out = self.transformer_encoder(x)

            out = out[:, 0]

            return out, sampled_mask
        else:
            return self.forward_eval(init_feats, init_weights)

    def forward_eval(self, init_feats: torch.Tensor, init_weights: torch.Tensor) -> torch.Tensor:
        feats = init_feats.permute(0, 2, 3, 1).reshape(-1, 1, self.feats_channels)
        init_weights = init_weights.view(1, -1, self.feats_channels)

        x = torch.cat((feats, init_weights.repeat(feats.shape[0], 1, 1)), dim=1)

        out = self.transformer_encoder(x)

        out = out[:, 0].view(init_feats.shape[0], init_feats.shape[2], init_feats.shape[3], -1).permute(0, 3, 1, 2)

        return out


class VCSegmentor(TSSegmentor):
    """
    Teacher-Student segmentor for semi-supervised learning with VC.
    """

    def __init__(self, cfg, segmentor_class: Callable[..., GenericModule]):
        super(VCSegmentor, self).__init__(cfg, segmentor_class)

        self._vc_loss = VCLoss(type="ce")

        self.feats_s: Optional[torch.Tensor] = None
        self.feats_t: Optional[torch.Tensor] = None

        self.segmentor.classifier.register_forward_hook(self._hook_student_feats)
        self.ema_segmentor.classifier.register_forward_hook(self._hook_teacher_feats)

        self.weight_generator = WeightGenerator(feats_channels=256)

    def _hook_student_feats(self, hooked_module, input, output):
        self.feats_s = input[0]

    def _hook_teacher_feats(self, hooked_module, input, output):
        self.feats_t = input[0].detach()

    @torch.no_grad()
    def _pseudo_labeling(self, data_list: TDataList, sharpen_factor: float = 0.5) -> torch.Tensor:
        # Update momentum batch statistic pl
        self.switch_to_bn_statistics_pl()
        self.ema_segmentor.train()
        self.forward_generic(data_list, selector="teacher")
        self.ema_segmentor.eval()

        # Pseudo labeling
        logits = self.forward_generic(data_list, selector="teacher")
        pseudo_probs = F.softmax(logits, dim=1)
        if sharpen_factor > 0:
            pseudo_probs = sharpen(pseudo_probs, sharpen_factor)

        self.switch_to_bn_statistics_eval()

        # Potetial labeling
        self.segmentor.eval()
        data_list_flipped = deepcopy(data_list)
        for data in data_list_flipped:
            data.img = torch.flip(data.img, dims=(2,))

        logits = self.forward_generic(data_list_flipped, selector="student")
        logits = torch.flip(logits, dims=(3,))
        potential_probs = F.softmax(logits, dim=1)
        if sharpen_factor > 0:
            potential_probs = sharpen(potential_probs, sharpen_factor)

        self.segmentor.train()

        return pseudo_probs.detach(), potential_probs.detach()

    def forward_train(
        self,
        data_list_l_weak: TDataList,
        data_list_u_weak: TDataList,
        data_list_l_strong: TDataList,
        data_list_u_strong: TDataList,
    ):
        # 1. Train with labeled data
        logits_l = self.forward_generic(data_list_l_strong, selector="student")
        labels_l = torch.cat([data.label for data in data_list_l_weak], dim=0).long()
        vc_weights_l, vc_weights_l_sampled_mask = self.weight_generator(
            self.feats_s.detach(), self.segmentor.classifier.weight.detach(), labels_l, torch.max(logits_l, dim=1)[0]
        )
        with torch.no_grad():
            vc_weights_l_target = (
                self.segmentor.classifier.weight.detach()[
                    F.interpolate(labels_l.unsqueeze(1).float(), size=self.feats_s.shape[2:4], mode="nearest")
                    .long()
                    .clamp_(max=logits_l.shape[1] - 1)
                    .view(-1)
                ]
                .squeeze(2)
                .squeeze(2)
            )
            vc_weights_l_target = vc_weights_l_target[vc_weights_l_sampled_mask.view(-1)]

        loss_dict_l = self.get_losses(
            logits=logits_l,
            labels=labels_l,
            gt_labels=labels_l,
            vc_weights=vc_weights_l,
            vc_weights_target=vc_weights_l_target,
            vc_weights_gt_target=vc_weights_l_target,
            labelled=True,
        )

        # 2. Pseudo labeling and Consistency training with labeled and unlabeled data
        logits_u = self.forward_generic(data_list_u_strong, selector="student")
        vc_weights_u = self.weight_generator(self.feats_s.detach(), self.segmentor.classifier.weight.detach())

        with torch.no_grad():
            if self.cfg.solver.vc_weight_norm == "adaptive":
                w_norm = self.segmentor.classifier.weight.norm(dim=1).mean().detach()
            if self.cfg.solver.vc_weight_norm == "adaptive_min":
                w_norm = self.segmentor.classifier.weight.norm(dim=1).min().detach()
            else:
                w_norm = 1 / self.cfg.solver.vc_weight_norm

        vc_weights_u = F.normalize(vc_weights_u, dim=1) * w_norm

        vc_logits_u = torch.einsum("bchw,bchw->bhw", self.feats_s, vc_weights_u.detach()).unsqueeze(1)
        vc_logits_u = F.interpolate(vc_logits_u, size=logits_u.shape[2:], mode="bilinear", align_corners=False)

        logits_u = torch.cat((logits_u, vc_logits_u), dim=1)

        pseudo_labels_u, potential_labels_u = self._pseudo_labeling(data_list_u_weak)
        labels_u = torch.cat([data.label for data in data_list_u_weak], dim=0).long()
        with torch.no_grad():
            vc_weights_u_target = (
                self.segmentor.classifier.weight.detach()[
                    F.interpolate(
                        torch.argmax(pseudo_labels_u, dim=1).unsqueeze(1).float(),
                        size=self.feats_s.shape[2:4],
                        mode="nearest",
                    )
                    .long()
                    .clamp_(max=self.cfg.model.num_classes - 1)
                    .squeeze(1)
                ]
                .squeeze(4)
                .squeeze(4)
                .permute(0, 3, 1, 2)
            )

            vc_weights_u_gt_target = (
                self.segmentor.classifier.weight.detach()[
                    F.interpolate(labels_u.unsqueeze(1).float(), size=self.feats_s.shape[2:4], mode="nearest")
                    .long()
                    .clamp_(max=self.cfg.model.num_classes - 1)
                    .squeeze(1)
                ]
                .squeeze(4)
                .squeeze(4)
                .permute(0, 3, 1, 2)
            )

        loss_dict_u, vc_stat_dict = self.get_losses(
            logits=logits_u,
            labels=pseudo_labels_u,
            gt_labels=labels_u,
            potential_labels=potential_labels_u,
            vc_weights=vc_weights_u,
            vc_weights_target=vc_weights_u_target,
            vc_weights_gt_target=vc_weights_u_gt_target,
            labelled=False,
        )

        loss_dict = Dict()
        loss_dict.loss_l = loss_dict_l
        loss_dict.loss_u = loss_dict_u

        # 3. Calculate mIoU of pseudo labels
        stat_dict = self._pseudo_labeling_statistics(pseudo_labels_u, labels_u)
        stat_dict.update(vc_stat_dict)

        return loss_dict, stat_dict

    def get_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        gt_labels: Optional[torch.Tensor] = None,
        potential_labels: Optional[torch.Tensor] = None,
        vc_weights: Optional[torch.Tensor] = None,
        vc_weights_target: Optional[torch.Tensor] = None,
        vc_weights_gt_target: Optional[torch.Tensor] = None,
        labelled: bool = True,
    ) -> Dict:
        loss_dict = Dict()
        if labelled:
            loss = F.cross_entropy(logits, labels.long(), ignore_index=255)

            loss_vw = -F.cosine_similarity(vc_weights, vc_weights_target).mean()

            loss_dict.loss = loss
            loss_dict.loss_vw = loss_vw

            return loss_dict
        else:
            pseudo_probs_u, pseudo_preds_u = torch.max(labels, dim=1)
            mask = torch.ge(pseudo_probs_u, self.cfg.model.teacher.low_score_threshold)
            valid_mask = torch.ne(gt_labels, 255) & mask

            _, p_labels1 = torch.max(labels, dim=1)
            _, p_labels2 = torch.max(potential_labels, dim=1)

            low_confidence_mask = torch.lt(pseudo_probs_u, self.cfg.model.teacher.score_threshold) & mask

            p_labels2 = torch.where(low_confidence_mask, torch.sort(labels, dim=1, descending=True)[1][:, 1], p_labels2)

            vc_mask = p_labels1 != p_labels2
            vc_mask_weight = torch.ones_like(vc_mask).float()
            vc_mask_weight.masked_fill_(vc_mask, 3)

            loss = self._vc_loss(logits, p_labels1, p_labels2, reduction="none")
            loss = loss * valid_mask.float() * vc_mask_weight
            loss = loss.sum() / (valid_mask.sum() + 1e-10)

            vc_ratio = (vc_mask.float() * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-10)

            with torch.no_grad():
                vcw_mask = F.interpolate(
                    (valid_mask & ~vc_mask).unsqueeze(1).float(), size=vc_weights.shape[2:4], mode="nearest",
                ).squeeze(1)

            loss_vw = (-F.cosine_similarity(vc_weights, vc_weights_target) * vcw_mask.float()).sum() / (
                vcw_mask.float().sum() + 1e-10
            )

            with torch.no_grad():
                vcw_mask = F.interpolate(
                    (valid_mask & vc_mask).unsqueeze(1).float(), size=vc_weights.shape[2:4], mode="nearest",
                ).squeeze(1)

                sim_matrix = torch.cosine_similarity(vc_weights, vc_weights_gt_target, dim=1)
                vc_weights_sim = (sim_matrix * vcw_mask.float()).sum() / (vcw_mask.float().sum() + 1e-10)

            loss_dict.loss = loss
            loss_dict.loss_vw = loss_vw

            return loss_dict, Dict(vc_weights_sim=vc_weights_sim, vc_ratio=vc_ratio)


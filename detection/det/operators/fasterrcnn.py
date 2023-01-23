from torch.nn.parallel import DistributedDataParallel
from torchvision.ops import clip_boxes_to_image

import gvcore.utils.distributed as dist_utils
from gvcore.utils.misc import sum_loss
from gvcore.operators import OPERATOR_REGISTRY, GenericOpt

from model.detector.fasterrcnn import FasterRCNN


@OPERATOR_REGISTRY.register("fasterrcnn")
class FasterRCNNOpt(GenericOpt):
    def __init__(self, cfg):
        super(FasterRCNNOpt, self).__init__(cfg)

    def build_model(self):
        model = FasterRCNN(self.cfg)
        model.to(self.device)
        if dist_utils.is_distributed():
            model = DistributedDataParallel(model)
        self.model = model

    def train_run_step(self):
        self.optimizer.zero_grad()
        data_list = self.train_loader.get_batch("cuda")
        loss_dict = self.model(data_list)
        losses = sum_loss(loss_dict)
        losses.backward()
        self.optimizer.step()

        metrics = {"lr": self.lr_scheduler.get_last_lr()[0]}
        metrics.update(loss_dict)
        self.summary.update(metrics)

        self.lr_scheduler.step()

    def test_run_step(self):
        data_list = self.test_loader.get_batch("cuda")
        if data_list is None:
            return True

        detected_bboxes = self.model(data_list)

        for data, detected_bbox in zip(data_list, detected_bboxes):
            meta = data.meta
            scale_x, scale_y = (
                1.0 * meta.ori_size[1] / meta.cur_size[1],
                1.0 * meta.ori_size[0] / meta.cur_size[0],
            )
            detected_bbox[:, [0, 2]] *= scale_x
            detected_bbox[:, [1, 3]] *= scale_y
            detected_bbox[:, :4] = clip_boxes_to_image(detected_bbox[:, :4], meta.ori_size)

            self.evaluator.process(data.meta.id, detected_bbox)
        return False

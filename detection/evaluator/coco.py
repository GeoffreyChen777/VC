# Copyright (c) Facebook, Inc. and its affiliates.
import os
import itertools
import numpy as np
import torch
from tabulate import tabulate
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from evaluator.fast_coco_api import COCOevalFast

from utils.meta import COCOMeta
from utils.box import xyxy2ltxywh

from gvcore.utils.logger import logger
import gvcore.utils.distributed as dist_utils
from gvcore.evaluator import EVALUATOR_REGISTRY


@EVALUATOR_REGISTRY.register("coco")
class COCOEvaluator:
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def __init__(self, json_file, distributed=False, use_fast_impl=True):
        """
        Args:
            distributed (True): if True, will collect results from all ranks and run evaluation
                in the main process.
                Otherwise, will only evaluate the results in the current process.
            use_fast_impl (bool): use a fast but **unofficial** implementation to compute AP.
                Although the results should be very close to the official implementation in COCO
                API, it is still recommended to compute results with the official API for use in
                papers. The faster implementation also uses more RAM.
        """
        self._distributed = distributed
        self._use_fast_impl = use_fast_impl

        self.coco_meta = COCOMeta()

        self._predictions = []

        if dist_utils.is_main_process():
            self._coco_api = COCO(json_file)

    def reset(self):
        self._predictions = []

    def process(self, img_id, output):
        prediction = self.output_to_coco_json(output, img_id)
        self._predictions.extend(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            dist_utils.synchronize()
            gathered_predictions = dist_utils.gather(self._predictions, dst=0)
            predictions = []
            for prediction in gathered_predictions:
                predictions.extend(prediction)
            if not dist_utils.is_main_process():
                return [None] * 12
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return

        file_path = os.path.join("./cache/predictions.pth")
        torch.save(predictions, file_path)

        logger.info("Preparing results for COCO format ...")
        logger.info(
            "Evaluating predictions with {} COCO API...".format("detectron2" if self._use_fast_impl else "official")
        )

        coco_dt = self._coco_api.loadRes(predictions)
        coco_eval = (COCOevalFast if self._use_fast_impl else COCOeval)(self._coco_api, coco_dt, "bbox")
        if img_ids is not None:
            coco_eval.params.imgIds = img_ids

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        self._per_cat_results(coco_eval, [meta["name"] for meta in self.coco_meta.meta][:-1])

        return coco_eval.stats

    def _per_cat_results(self, coco_eval, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.
        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.
        Returns:
            a dict of {metric name: score}
        """
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d, tablefmt="pipe", floatfmt=".3f", headers=["category", "AP"] * (N_COLS // 2), numalign="left",
        )
        logger.info("Per-category mAP: \n" + table)

    def output_to_coco_json(self, outputs, img_id):
        """
        Dump an "Instances" object to a COCO-format json that's used for evaluation.

        Args:
            outputs (torch.Tensor):
            img_id (int): the image id

        Returns:
            list[dict]: list of json annotations in COCO format.
        """
        results = []
        outputs[:, :4] = xyxy2ltxywh(outputs[:, :4])
        for output in outputs:
            result = {
                "image_id": img_id,
                "category_id": self.coco_meta.coco80to91(int(output[4])),
                "bbox": [x.item() for x in output[:4]],
                "score": output[5].item(),
            }
            results.append(result)
        return results

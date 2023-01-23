# Copyright (c) Facebook, Inc. and its affiliates.

from gvcore.evaluator import EVALUATOR_REGISTRY
from utils.meta import VOCMeta
from utils.box import xyxy2ltxywh
from evaluator.coco import COCOEvaluator


@EVALUATOR_REGISTRY.register("voc")
class VOCEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.

    COCO style for VOC
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
        super(VOCEvaluator, self).__init__(json_file, distributed, use_fast_impl)
        self._meta = VOCMeta()

    def _per_cat_results(self, coco_eval, class_names=None):
        pass

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
                "category_id": int(output[4]),
                "bbox": [x.item() for x in output[:4]],
                "score": output[5].item(),
            }
            results.append(result)
        return results

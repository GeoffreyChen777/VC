from gvcore.dataset import DATASET_REGISTRY

from dataset.det_generic import DetGenericDataset
from utils.meta import cocometa


@DATASET_REGISTRY.register("COCO")
class COCODataset(DetGenericDataset):
    def __init__(self, cache, path, img_transforms, subset):
        super(COCODataset, self).__init__(cache, path, img_transforms, cocometa, subset)

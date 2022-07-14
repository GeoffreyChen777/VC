from gvcore.dataset import DATASET_REGISTRY

from dataset.det_generic import DetGenericDataset
from utils.meta import vocmeta


@DATASET_REGISTRY.register("VOC")
class VOCDataset(DetGenericDataset):
    def __init__(self, cache, path, img_transforms, subset):
        super(VOCDataset, self).__init__(cache, path, img_transforms, vocmeta, subset)

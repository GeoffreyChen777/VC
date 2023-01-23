from typing import Callable, Optional
import torchvision.io as io

from gvcore.utils.structure import GenericData
from gvcore.dataset import DATASET_REGISTRY, GenericDataset


@DATASET_REGISTRY.register("VOC")
class VOCDataset(GenericDataset):
    def __init__(
        self, root: str, subset: str = "train", img_transforms: Optional[Callable[[GenericData], GenericData]] = None
    ):
        super(VOCDataset, self).__init__(root, subset, img_transforms)

    def _get_data_ids(self):
        with open(f"{self._root}/index/{self._subset}.txt", "r") as f:
            data_ids = [line.strip() for line in f.readlines()]

        return sorted(data_ids)

    def _load(self, item_idx):
        data_id = self._data_ids[item_idx]
        img = io.read_image(f"{self._root}/image/{data_id}.jpg")
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        label = io.read_image(f"{self._root}/label/{data_id}.png")

        data = GenericData()
        data.set("img", img)
        data.set("label", label)

        data.set("meta", GenericData())
        data.meta.set("id", data_id)

        img_size = data.img.shape[1:]
        data.meta.set("ori_size", list(img_size))
        data.meta.set("cur_size", list(img_size))

        return data

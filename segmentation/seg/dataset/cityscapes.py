import os
from typing import Callable, Optional
import torchvision.io as io

from gvcore.utils.distributed import is_main_process, synchronize
from gvcore.utils.structure import GenericData
from gvcore.dataset import DATASET_REGISTRY, GenericDataset


@DATASET_REGISTRY.register("Cityscapes")
class CityscapeDataset(GenericDataset):
    def __init__(
        self,
        root: str = "./data/cityscapes",
        subset: str = "train",
        img_transforms: Optional[Callable[[GenericData], GenericData]] = None,
    ):
        super(CityscapeDataset, self).__init__(root, subset, img_transforms)

    def _get_data_ids(self):
        # 0. Create index file

        if is_main_process():
            if not os.path.exists(f"{self._root}/index/{self._subset}.txt"):
                os.makedirs(f"{self._root}/index", exist_ok=True)
                with open(f"{self._root}/index/{self._subset}.txt", "w") as w:
                    for p, d, f in os.walk(f"{self._root}/leftImg8bit/{self._subset}"):
                        for file in f:
                            if file.endswith(".png"):
                                file_path = os.path.join(p, file)
                                file_idx = file_path.replace("_leftImg8bit.png", "").replace(
                                    f"{self._root}/leftImg8bit/", ""
                                )
                                w.write(f"{file_idx}\n")
        synchronize()

        # 1. Read index file
        with open(f"{self._root}/index/{self._subset}.txt", "r") as f:
            data_ids = [line.strip() for line in f.readlines()]

        return sorted(data_ids)

    def _load(self, item_idx):
        data_id = self._data_ids[item_idx]
        img = io.read_image(f"{self._root}/leftImg8bit/{data_id}_leftImg8bit.png")
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        label = io.read_image(f"{self._root}/gtFine/{data_id}_gtFine_labelTrainIds.png")

        data = GenericData()
        data.set("img", img)
        data.set("label", label)

        data.set("meta", GenericData())
        data.meta.set("id", data_id)

        img_size = data.img.shape[1:]
        data.meta.set("ori_size", list(img_size))
        data.meta.set("cur_size", list(img_size))

        return data


color_map = [
    (128, 64, 128),
    (244, 35, 232),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

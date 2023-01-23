from typing import Callable, Optional
import torch
from gvcore.utils.structure import GenericData
from gvcore.utils.registry import Registry


DATASET_REGISTRY = Registry()


class GenericDataset(torch.utils.data.Dataset):
    def __init__(
        self, root: str, subset: str = "train", img_transforms: Optional[Callable[[GenericData], GenericData]] = None
    ):
        super(GenericDataset, self).__init__()

        self._root = root
        self._subset = subset
        self._img_transforms = img_transforms

        self._data_ids = self._get_data_ids()

    def _get_data_ids(self):
        raise NotImplementedError

    def __len__(self):
        return len(self._data_ids)

    def _load(self, item_idx: int) -> GenericData:
        raise NotImplementedError

    def __getitem__(self, item_idx):
        sample = self._load(item_idx)
        if self._img_transforms is not None:
            sample = self._img_transforms(sample)
        return sample

    @staticmethod
    def collate_fn(batch):
        return batch

import torch
import torchvision.io as io
from gvcore.utils.structure import GenericData
from gvcore.utils.registry import Registry


DATASET_REGISTRY = Registry()
TRANSFORM_REGISTRY = Registry()


class GenericDataset(torch.utils.data.Dataset):
    def __init__(self, cache, path, img_transforms, meta=None, subset=None):
        super(GenericDataset, self).__init__()
        assert cache is not None or path is not None, "No data cache or data path specified."

        self._cache = cache
        self._path = path

        self._data_ids = self._get_data_ids()

        self._img_transforms = img_transforms

        self._meta = meta

    def _get_data_ids(self):
        if self._cache is not None:
            return sorted(list(self._cache.keys()))
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self._data_ids)

    def _load(self, item_idx):
        data_id = self._data_ids[item_idx]
        cache = self._cache[data_id]

        data = GenericData()

        data.set("img", io.read_image(cache["image_path"]))
        if data.img.shape[0] == 1:
            data.img = data.img.repeat(3, 1, 1)
        label = torch.from_numpy(cache["label"])
        data.set("label", label)

        data.set("meta", GenericData())
        data.meta.set("id", data_id)
        data.meta.set("image_path", cache["image_path"])

        img_size = data.img.shape[1:]
        data.meta.set("ori_size", list(img_size))
        data.meta.set("cur_size", list(img_size))
        data.meta.set("flip", False)

        return data

    def __getitem__(self, item_idx):
        sample = self._load(item_idx)
        if self._img_transforms is not None:
            sample = self._img_transforms(sample)
        return sample

    @staticmethod
    def collate_fn(batch):
        return batch

import numpy as np
from tabulate import tabulate

from gvcore.utils.logger import logger
from gvcore.dataset import GenericDataset

from utils.box import nonempty


class DetGenericDataset(GenericDataset):
    def __init__(self, cache, path, img_transforms, meta=None, subset=None):
        super(DetGenericDataset, self).__init__(cache, path, img_transforms, meta)

        if subset != "test":
            self._filter_empty()
        self.group = self._generate_group()

    def _statics(self):
        """
        To print the statistics for detection dataset such as MSCOCO.
        """
        if self._meta is None:
            return

        _categories_count = len(self._meta) - 1

        categories_count = np.zeros(_categories_count)
        for data_id, data in self._cache.items():
            category_label = data["label"][:, 4].astype(np.int64)
            category_label_count = np.bincount(category_label, minlength=_categories_count)
            categories_count += category_label_count
        table = []
        row = []
        for i, count in enumerate(categories_count):
            if i % 4 == 0:
                if len(row) > 0:
                    table.append(row)
                row = []
            row.append(self._meta(i)["name"])
            row.append(count)

        logger.info(
            "\n"
            + tabulate(
                table,
                headers=["Category", "Count", "Category", "Count", "Category", "Count", "Category", "Count"],
                tablefmt="fancy_grid",
            )
        )

    def _filter_empty(self):
        """
        To filter no label images and invalid labels
        """
        for data_id, data in self._cache.items():
            label = data["label"]
            keep = nonempty(label)
            data["label"] = label[keep]
            if data["label"].shape[0] == 0:
                self._data_ids.remove(data_id)

    def _generate_group(self):
        """
        To generate group of images according to aspect ratios.
        """
        group = []
        for data_id in self._data_ids:
            data = self._cache[data_id]
            group.append(data["aspect_ratio"])
        return group


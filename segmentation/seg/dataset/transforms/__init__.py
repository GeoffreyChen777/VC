from typing import Iterable
import torch
import torchvision.transforms.functional as tvF

from gvcore.dataset.transforms import TRANSFORM_REGISTRY
import gvcore.dataset.transforms as GVTransforms
from gvcore.utils.structure import GenericData


@TRANSFORM_REGISTRY.register("resize")
class Resize(GVTransforms.Resize):
    def _process_label(self, data: GenericData, params: any) -> GenericData:
        if data.has("label") and self.process_label:
            new_h, new_w = params[0], params[1]
            data.label = tvF.resize(data.label, size=[new_h, new_w], interpolation=tvF.InterpolationMode.NEAREST)
        return data


@TRANSFORM_REGISTRY.register("random_horizontal_flip")
class RandomHorizontalFlip(GVTransforms.RandomHorizontalFlip):
    def _process_label(self, data: GenericData, params: any) -> GenericData:
        if data.has("label") and params:
            data.label = tvF.hflip(data.label)
        return data


@TRANSFORM_REGISTRY.register("random_crop")
class RandomCrop(GVTransforms.RandomCrop):
    def _process_label(self, data: GenericData, params: any) -> GenericData:
        top, left, crop_h, crop_w = params
        data.label = tvF.crop(data.label, top, left, crop_h, crop_w)
        return data


@TRANSFORM_REGISTRY.register("batch_pad")
class BatchPad(GVTransforms.BatchPad):
    def __init__(self, fill=0, stride=32, label_fill=255, fixed_size=None):
        self.fill = fill
        self.label_fill = label_fill

        self.stride = stride
        self.fixed_size = fixed_size

    def _get_params(self, batch: Iterable[GenericData]):
        image_sizes = [data.img.shape[-2:] for data in batch]
        max_size = torch.tensor(image_sizes).max(0).values
        max_size = torch.div((max_size + (self.stride - 1)), self.stride, rounding_mode="trunc") * self.stride

        if self.fixed_size is not None:
            max_size[0] = max(max_size[0], self.fixed_size[0])
            max_size[1] = max(max_size[1], self.fixed_size[1])

        return max_size[0], max_size[1], image_sizes

    def _process_img(self, batch: Iterable[GenericData], params: any) -> Iterable[GenericData]:
        for i, (data, img_size) in enumerate(zip(batch, params[2])):
            pad_h = params[0] - img_size[0]
            pad_w = params[1] - img_size[1]
            data.img = tvF.pad(data.img, padding=[0, 0, pad_w, pad_h], fill=self.fill)
            data.meta.pad_size = [0, 0, int(pad_w), int(pad_h)]
            batch[i] = data
        return batch

    def _process_label(self, batch: Iterable[GenericData], params: any) -> Iterable[GenericData]:
        for i, (data, img_size) in enumerate(zip(batch, params[2])):
            pad_h = params[0] - img_size[0]
            pad_w = params[1] - img_size[1]
            data.label = tvF.pad(data.label, padding=[0, 0, pad_w, pad_h], fill=self.label_fill)
            batch[i] = data
        return batch


@TRANSFORM_REGISTRY.register("k_random_erase")
class RandomErase(GVTransforms.KRandomErase):
    def _process_label(self, data: GenericData, params: any) -> GenericData:
        if data.has("label") and params:
            params["values"].fill_(255.0)
            data.label = self.transform(data.label.float(), params=params).to(data.label.dtype)
        return data

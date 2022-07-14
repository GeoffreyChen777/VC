import torch
import numpy as np
import torchvision.transforms.functional as tvf
import torchvision.transforms as tv

from gvcore.dataset import TRANSFORM_REGISTRY
from gvcore.dataset.transforms import *

from dataset.transforms.functional import *


@TRANSFORM_REGISTRY.register("resize")
class Resize(Transforms):
    def __init__(self, min_size=(800,), max_size=1333, mode="choice"):
        self.min_size = min_size
        if isinstance(self.min_size, int):
            self.min_size = (self.min_size, self.min_size)
        self.max_size = max_size
        self.mode = mode

    def __call__(self, data):
        return resize(data, min_size=self.min_size, max_size=self.max_size, mode=self.mode)


@TRANSFORM_REGISTRY.register("normalize")
class Normalize(Transforms):
    def __init__(self, mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), scale=False):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, data):
        return normalise(data, mean=self.mean, std=self.std, scale=self.scale)


@TRANSFORM_REGISTRY.register("denormalize")
class Denormalize(Transforms):
    def __init__(self, mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), scale=False):
        self.mean = mean
        self.std = std
        self.scale = scale

    def __call__(self, data):
        return denormalise(data, mean=self.mean, std=self.std, scale=self.scale)


@TRANSFORM_REGISTRY.register("random_horizontal_flip")
class RandomHorizontalFlip(Transforms):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        return random_apply(data, transform=horizontal_flip, p=self.p)


@TRANSFORM_REGISTRY.register("random_erase")
class RandomErase(Transforms):
    def __init__(self, scale=(0.02, 0.2), ratio=(0.1, 6.0), p=0.5):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.func = tv.RandomErasing(p=p, scale=scale, ratio=ratio, value="random")

    def __call__(self, data):
        data.img = self.func(data.img)
        return data

    def __repr__(self):
        return f"{self.__class__.__name__}(p: {self.p}, scale: {self.scale}, ratio: {self.ratio})"


@TRANSFORM_REGISTRY.register("erase_within_box")
class EraseWithinBox(Transforms):
    def __init__(self, cut_num=1, cut_ratio=(0.1, 0.3), fill=(0, 0, 0)):
        assert isinstance(cut_ratio, tuple)
        assert cut_ratio[0] < cut_ratio[1]
        self.cut_num = cut_num
        self.cut_ratio = cut_ratio
        self.fill = torch.tensor(fill).float().view(-1, 1, 1)

    def __call__(self, data):
        h, w = data.meta.cur_size
        label_num = data.label.shape[0]
        sample_ratio = (
            torch.rand(size=(label_num, 2), device=data.label.device) * (self.cut_ratio[1] - self.cut_ratio[0])
            + self.cut_ratio[0]
        )

        box_w, box_h = (data.label[:, 2] - data.label[:, 0], data.label[:, 3] - data.label[:, 1])

        cutout_half_w, cutout_half_h = box_w * sample_ratio[:, 0] / 2.0, box_h * sample_ratio[:, 1] / 2.0

        sample_bias_ratio = torch.rand(size=(label_num, 2), device=data.label.device)
        cutout_x_bias, cutout_y_bias = box_w * sample_bias_ratio[:, 0], box_h * sample_bias_ratio[:, 1]
        cutout_x, cutout_y = data.label[:, 0] + cutout_x_bias, data.label[:, 1] + cutout_y_bias
        cutout_x1 = (cutout_x - cutout_half_w).clamp(min=0, max=w).long()
        cutout_y1 = (cutout_y - cutout_half_h).clamp(min=0, max=h).long()
        cutout_x2 = (cutout_x + cutout_half_w).clamp(min=0, max=w).long()
        cutout_y2 = (cutout_y + cutout_half_h).clamp(min=0, max=h).long()
        for i in range(label_num):
            data.img[:, cutout_y1[i] : cutout_y2[i], cutout_x1[i] : cutout_x2[i]] = self.fill

        return data


@TRANSFORM_REGISTRY.register("color_jitter")
class ColorJitter(Transforms):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.func = tv.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("grayscale")
class Grayscale(Transforms):
    def __init__(self, num_output_channels=3):
        self.func = tv.Grayscale(num_output_channels=num_output_channels)

    def __call__(self, data):
        data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("random_color_jitter")
class RandomColorJitter(Transforms):
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.p = p
        self.func = tv.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, data):
        data.img = random_apply(data.img, transform=self.func, p=self.p)
        return data


@TRANSFORM_REGISTRY.register("random_grayscale")
class RandomGrayscale(Transforms):
    def __init__(self, p=0.2, num_output_channels=3):
        self.p = p
        self.func = tv.Grayscale(num_output_channels=num_output_channels)

    def __call__(self, data):
        data.img = random_apply(data.img, transform=self.func, p=self.p)
        return data


@TRANSFORM_REGISTRY.register("random_gaussian_blur")
class RandomGaussianBlur(Transforms):
    def __init__(self, p=0.5, sigma=(0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def __call__(self, data):
        return random_apply(data, transform=gaussian_blur, p=self.p, sigma=self.sigma)


@TRANSFORM_REGISTRY.register("batch_apply")
class BatchApply(Transforms):
    def __init__(self, **transforms):
        self.transforms = parse_transform_config(transforms)

    def __call__(self, batch):
        for i, data in enumerate(batch):
            batch[i] = self.transforms(data)
        return batch

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        if isinstance(self.transforms, Compose):
            for t in self.transforms:
                format_string += "\n"
                format_string += "{0}".format(t)
        else:
            format_string += "{0}".format(self.transforms)
        format_string += "\n)"
        return format_string


@TRANSFORM_REGISTRY.register("batch_pad")
class BatchPad(Transforms):
    def __init__(self, fill=0, stride=32):
        self.fill = fill
        self.stride = stride

    def __call__(self, batch):
        image_sizes = [data.meta.cur_size for data in batch]
        max_size = torch.tensor(image_sizes).max(0).values
        max_size = torch.div((max_size + (self.stride - 1)), self.stride, rounding_mode="trunc") * self.stride

        for i, data in enumerate(batch):
            pad_h = max_size[0] - data.meta.cur_size[0]
            pad_w = max_size[1] - data.meta.cur_size[1]
            data.img = tvf.pad(data.img, padding=[0, 0, pad_w, pad_h], fill=self.fill)
            data.meta.pad_size = [0, 0, int(pad_w), int(pad_h)]
            batch[i] = data
        return batch


@TRANSFORM_REGISTRY.register("random_select")
class RandomSelect(Transforms):
    def __init__(self, num, transforms):
        self.num = num
        self.transforms = parse_transform_config(transforms)

    def __call__(self, data):
        idx = torch.randperm(len(self.transforms))[: self.num].sort()[0]
        for i in idx:
            data = self.transforms[i](data)
        return data


@TRANSFORM_REGISTRY.register("repeat")
class Repeat(Transforms):
    def __init__(self, **transforms):
        self.transforms = []
        for name, args in transforms.items():
            for arg in args:
                self.transforms.append(parse_transform_config({name: arg}))
        self.transforms = Compose(self.transforms)

    def __call__(self, data):
        return self.transforms(data)

    def __repr__(self):
        trans_strs = []
        for transform in self.transforms:
            trans_strs.append(str(transform))
        trans_strs = ", ".join(trans_strs)
        return f"{self.__class__.__name__}({trans_strs})"


@TRANSFORM_REGISTRY.register("random_adjust")
class RandomAdjust(Transforms):
    def __init__(self, adjust_func, adjust_range=(0.5, 1.5)):
        if not isinstance(adjust_range, (tuple, list)):
            adjust_range = (adjust_range, adjust_range)
        assert adjust_range[0] <= adjust_range[1]
        self.adjust_range = adjust_range
        self.adjust_func = adjust_func

    def __call__(self, data):
        adjust_factor = (
            torch.rand((1,), device=data.img.device) * (self.adjust_range[1] - self.adjust_range[0])
            + self.adjust_range[0]
        )
        data.img = self.adjust_func(data.img, adjust_factor)
        return data


@TRANSFORM_REGISTRY.register("random_brightness")
class RandomBrightness(RandomAdjust):
    def __init__(self, brightness_range=(0.5, 1.5)):
        super(RandomBrightness, self).__init__(tvf.adjust_brightness, brightness_range)


@TRANSFORM_REGISTRY.register("random_hue")
class RandomHue(RandomAdjust):
    def __init__(self, hue_range=(-0.25, 0.25)):
        super(RandomHue, self).__init__(tvf.adjust_hue, hue_range)


@TRANSFORM_REGISTRY.register("random_contrast")
class RandomContrast(RandomAdjust):
    def __init__(self, contrast_range=(0.5, 1.5)):
        super(RandomContrast, self).__init__(tvf.adjust_contrast, contrast_range)


@TRANSFORM_REGISTRY.register("random_saturation")
class RandomSaturation(RandomAdjust):
    def __init__(self, saturation_range=(0.5, 1.5)):
        super(RandomSaturation, self).__init__(tvf.adjust_saturation, saturation_range)


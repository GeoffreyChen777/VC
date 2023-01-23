from typing import Callable, Iterable
from copy import deepcopy
import numpy as np
import torch
import torchvision.transforms as tvT
import torchvision.transforms.functional as tvF

from gvcore.utils.structure import GenericData
from gvcore.utils.registry import Registry

TRANSFORM_REGISTRY = Registry()


def parse_transform_config(cfg):
    transforms = []
    if cfg is not None:
        for name, args in cfg.items():
            if args is not None:
                if isinstance(args, (tuple, list)):
                    transforms.append(TRANSFORM_REGISTRY[name](*args))
                elif isinstance(args, dict):
                    transforms.append(TRANSFORM_REGISTRY[name](**args))
                else:
                    raise ValueError(f"Argument for transform {name} is invalid, got {args}")
            else:
                transforms.append(TRANSFORM_REGISTRY[name]())
        if len(transforms) > 1:
            transforms = Compose(transforms)
        else:
            transforms = transforms[0]
    else:
        transforms = None
    return transforms


class Transforms:
    def _get_params(self, data: GenericData):
        return None

    @staticmethod
    def _process_img(data: GenericData, params: any) -> GenericData:
        raise NotImplementedError

    @staticmethod
    def _process_label(data: GenericData, params: any) -> GenericData:
        return data

    def __call__(self, data: GenericData) -> GenericData:
        params = self._get_params(data)
        data = self._process_img(data, params)
        data = self._process_label(data, params)
        return data

    def __repr__(self):
        params = []
        for k, v in self.__dict__.items():
            params.append(f"{k}: {v}")
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"


class DataCopy(Transforms):
    def __init__(self, transforms: Callable[..., GenericData]):
        self.transforms = transforms

    def __call__(self, data: GenericData) -> GenericData:
        data = deepcopy(data)
        data = self.transforms(data)
        return data


class Compose(Transforms):
    def __init__(self, transforms: Iterable[Transforms]):
        self.transforms = transforms

    def __call__(self, data: GenericData) -> GenericData:
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

    def __getitem__(self, item):
        return self.transforms[item]

    def __len__(self):
        return len(self.transforms)


@TRANSFORM_REGISTRY.register("identity")
class Identity(Transforms):
    def __call__(self, data: GenericData) -> GenericData:
        return data


@TRANSFORM_REGISTRY.register("normalize")
class Normalize(Transforms):
    def __init__(self, mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), scale=False, RGB2BGR=True):
        self.mean = mean
        self.std = std
        self.scale = scale
        self.RGB2BGR = RGB2BGR

    def _get_params(self, data: GenericData):
        return self.mean, self.std, self.scale, self.RGB2BGR

    @staticmethod
    def _process_img(data: GenericData, params: any) -> GenericData:
        mean, std, scale, RGB2BGR = params
        if RGB2BGR:
            data.img = data.img[[2, 1, 0], :, :]
        data.img = data.img.float()
        if scale:
            data.img = data.img / 255.0
        if data.img.shape[0] == 1 and data.img.dim() == 3:
            data.img = data.img.repeat(3, 1, 1)
        data.img = tvF.normalize(data.img, mean, std)
        return data


@TRANSFORM_REGISTRY.register("uint2float")
class UInt2Float(Transforms):
    @staticmethod
    def _process_img(data: GenericData, params: any) -> GenericData:
        if data.img.dtype == torch.uint8:
            data.img = data.img.float()
            data.img = data.img / 255.0
        return data



@TRANSFORM_REGISTRY.register("denormalize")
class Denormalize(Transforms):
    def __init__(self, mean=(103.53, 116.28, 123.675), std=(1.0, 1.0, 1.0), scale=False, RGB2BGR=True):
        self.mean = mean
        self.std = std
        self.scale = scale
        self.RGB2BGR = RGB2BGR

    def _get_params(self, data: GenericData):
        return self.mean, self.std, self.scale, self.RGB2BGR

    @staticmethod
    def _process_img(data: GenericData, params: any) -> GenericData:
        mean, std, scale, RGB2BGR = params
        data.img = tvF.normalize(data.img, mean=[0.0, 0.0, 0.0], std=[1 / x for x in std])
        data.img = tvF.normalize(data.img, mean=[-1 * x for x in mean], std=[1.0, 1.0, 1.0])
        if RGB2BGR:
            data.img = data.img[[2, 1, 0], :, :]
        if scale:
            data.img = data.img * 255.0
        return data


@TRANSFORM_REGISTRY.register("resize")
class Resize(Transforms):
    def __init__(self, min_size=(800,), max_size=1333, scale=None, mode="choice", process_label=True):
        self.min_size = min_size
        if isinstance(self.min_size, int):
            self.min_size = (self.min_size, self.min_size)
        self.max_size = max_size
        self.scale = scale
        if isinstance(self.scale, float):
            self.scale = (self.scale,)
        self.mode = mode
        self.process_label = process_label

    def _get_params(self, data: GenericData):
        h, w = data.img.shape[1:]
        if self.scale is None:
            if self.mode == "choice":
                size = np.random.choice(self.min_size)
            elif self.mode == "range":
                size = np.random.randint(min(self.min_size), max(self.min_size) + 1)
            else:
                raise ValueError("Unknown mode {}".format(self.mode))

            scale = size * 1.0 / min(h, w)
            if h < w:
                new_h, new_w = size, scale * w
            else:
                new_h, new_w = scale * h, size
            if max(new_h, new_w) > self.max_size:
                scale = self.max_size * 1.0 / max(new_h, new_w)
                new_h = new_h * scale
                new_w = new_w * scale
            new_w = int(new_w + 0.5)
            new_h = int(new_h + 0.5)
        else:
            sampled_scale = np.random.choice(self.scale)
            new_h = int(sampled_scale * h + 0.5)
            new_w = int(sampled_scale * w + 0.5)

        return (new_h, new_w, new_h / h, new_w / w)

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        data.img = tvF.resize(data.img, size=[params[0], params[1]])
        resized_size = list(data.img.shape[-2:])
        data.meta.cur_size = resized_size

        return data


class RandomTransforms(Transforms):
    def __init__(self, p: float = 0.5):
        self.p = p

    def _get_params(self, data: GenericData) -> bool:
        return torch.rand(1) <= self.p


@TRANSFORM_REGISTRY.register("random_horizontal_flip")
class RandomHorizontalFlip(RandomTransforms):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        if params:
            data.img = tvF.hflip(data.img)
            data.meta.flip = True
        else:
            data.meta.flip = False
        return data


@TRANSFORM_REGISTRY.register("random_erase")
class RandomErase(Transforms):
    def __init__(self, scale=(0.02, 0.2), ratio=(0.1, 6.0), p=0.5):
        self.func = tvT.RandomErasing(p=p, scale=scale, ratio=ratio, value="random")

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("color_jitter")
class ColorJitter(Transforms):
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.func = tvT.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("grayscale")
class Grayscale(Transforms):
    def __init__(self, num_output_channels=3):
        self.func = tvT.Grayscale(num_output_channels=num_output_channels)

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("random_color_jitter")
class RandomColorJitter(RandomTransforms):
    def __init__(self, p=0.8, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1):
        self.p = p
        self.func = tvT.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        if params:
            data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("random_grayscale")
class RandomGrayscale(RandomTransforms):
    def __init__(self, p=0.2, num_output_channels=3):
        self.p = p
        self.func = tvT.Grayscale(num_output_channels=num_output_channels)

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        if params:
            data.img = self.func(data.img)
        return data


@TRANSFORM_REGISTRY.register("random_gaussian_blur")
class RandomGaussianBlur(RandomTransforms):
    def __init__(self, p=0.5, sigma=(0.1, 2.0)):
        self.p = p
        self.sigma = sigma

    def _get_params(self, data: GenericData):
        kx = int(0.1 * data.meta.cur_size[1]) // 2 * 2 + 1
        ky = int(0.1 * data.meta.cur_size[0]) // 2 * 2 + 1
        return super()._get_params(data), kx, ky, self.sigma

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        if params[0]:
            data.img = tvF.gaussian_blur(data.img, [params[1], params[2]], params[3])
        return data


@TRANSFORM_REGISTRY.register("random_crop")
class RandomCrop(Transforms):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant"):
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _get_params(self, data: GenericData):
        crop_w, crop_h = self.size
        h, w = data.img.shape[-2:]

        if self.padding is not None:
            if isinstance(self.padding, int):
                padding = [self.padding, self.padding, self.padding, self.padding]
            elif isinstance(self.padding, (tuple, list)) and len(self.padding) == 2:
                padding = [self.padding[1], self.padding[1], self.padding[0], self.padding[0]]
            elif isinstance(self.padding, (tuple, list)) and len(self.padding) == 4:
                padding = [self.padding[3], self.padding[2], self.padding[1], self.padding[0]]
            else:
                raise ValueError("padding should be an integer, a 2-tuple, or a 4-tuple")
            data.img = tvF.pad(data.img, padding, self.fill, self.padding_mode)

        data.meta.cur_size = data.img.shape[-2:]
        h, w = data.img.shape[-2:]

        # pad the width if needed
        if self.pad_if_needed and w < self.size[1]:
            padding = [self.size[1] - w, 0]
            data.img = tvF.pad(data.img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and h < self.size[0]:
            padding = [0, self.size[0] - h]
            data.img = tvF.pad(data.img, padding, self.fill, self.padding_mode)

        data.meta.cur_size = data.img.shape[-2:]

        h, w = data.img.shape[-2:]
        if h < crop_h:
            top = 0
            crop_h = h
        else:
            top = torch.randint(0, h - crop_h + 1, (1,))

        if w < crop_w:
            left = 0
            crop_w = w
        else:
            left = torch.randint(0, w - crop_w + 1, (1,))

        return top, left, crop_h, crop_w

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        top, left, crop_h, crop_w = params
        data.img = tvF.crop(data.img, top, left, crop_h, crop_w)
        data.meta.cur_size = (crop_h, crop_w)
        return data


@TRANSFORM_REGISTRY.register("random_adjust")
class RandomAdjust(Transforms):
    def __init__(self, adjust_func, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10, factor_type="float"):
        self.adjust_func = adjust_func
        self.magnitude_limit = magnitude_limit
        self.random_max = random_max
        self.magnitude_bias = magnitude_bias
        self.factor_type = factor_type

    def _get_params(self, data: GenericData):
        random_factor = np.random.randint(1, self.random_max) / self.random_max
        adjust_factor = float(self.magnitude_limit) * random_factor
        if self.factor_type == "int":
            adjust_factor = int(adjust_factor)
        adjust_factor += self.magnitude_bias

        return adjust_factor

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        data.img = self.adjust_func(data.img, params)
        return data


@TRANSFORM_REGISTRY.register("random_brightness")
class RandomBrightness(RandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(RandomBrightness, self).__init__(tvF.adjust_brightness, magnitude_limit, magnitude_bias, random_max)


@TRANSFORM_REGISTRY.register("random_contrast")
class RandomContrast(RandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(RandomContrast, self).__init__(tvF.adjust_contrast, magnitude_limit, magnitude_bias, random_max)


@TRANSFORM_REGISTRY.register("random_saturation")
class RandomSaturation(RandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(RandomSaturation, self).__init__(
            tvF.adjust_saturation, magnitude_limit, magnitude_bias, random_max, "float"
        )


@TRANSFORM_REGISTRY.register("random_posterize")
class RandomPosterize(RandomAdjust):
    def __init__(self, magnitude_limit=4, magnitude_bias=4, random_max=10):
        super(RandomPosterize, self).__init__(tvF.posterize, magnitude_limit, magnitude_bias, random_max, "int")

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        if data.img.dtype != torch.float:
            data.img = tvF.posterize(data.img.contiguous(), params)
        else:
            data.img = tvF.posterize((data.img.contiguous() * 255.0).to(torch.uint8), params).float() / 255.0
        return data


@TRANSFORM_REGISTRY.register("random_sharpness")
class RandomSharpness(RandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(RandomSharpness, self).__init__(
            tvF.adjust_sharpness, magnitude_limit, magnitude_bias, random_max, "float"
        )


@TRANSFORM_REGISTRY.register("random_solarize")
class RandomSolarize(RandomAdjust):
    def __init__(self, magnitude_limit=1, magnitude_bias=0, random_max=10):
        super(RandomSolarize, self).__init__(tvF.solarize, magnitude_limit, magnitude_bias, random_max, "int")


@TRANSFORM_REGISTRY.register("autocontrast")
class AutoContrast(Transforms):
    def _process_img(self, data: GenericData, params: any) -> GenericData:
        img = data.img
        if img.dim() == 3:
            img.unsqueeze(0)

        B, C, H, W = img.shape

        x_min = img.view(B, C, -1).min(-1)[0].view(B, C, 1, 1)
        x_max = img.view(B, C, -1).max(-1)[0].view(B, C, 1, 1)

        data.img = ((img - x_min) / torch.clamp(x_max - x_min, min=1e-9, max=1)).expand_as(img)
        return data


@TRANSFORM_REGISTRY.register("equalize")
class Equalize(Transforms):
    def _process_img(self, data: GenericData, params: any) -> GenericData:
        if data.img.dtype != torch.float:
            data.img = tvF.equalize(data.img.contiguous())
        else:
            data.img = tvF.equalize((data.img.contiguous() * 255.0).to(torch.uint8)).float() / 255.0
        return data


@TRANSFORM_REGISTRY.register("random_rotate")
class RandomRotate(Transforms):
    def __init__(self, angle=(-30, 30)):
        if isinstance(angle, int) or isinstance(angle, float):
            angle = (angle, angle)
        self.angle = angle

    def _get_params(self, data: GenericData):
        return np.random.uniform(self.angle[0], self.angle[1])

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        data.img = tvF.rotate(data.img, angle=params)
        return data


@TRANSFORM_REGISTRY.register("random_shear")
class RandomShear(Transforms):
    def __init__(self, angle=(-17, 17), direction="random"):
        self.angle = angle
        self.direction = direction

    def _get_params(self, data: GenericData):
        if self.direction == "random":
            direction = np.random.choice(["horizontal", "vertical"])
        else:
            direction = self.direction

        return np.random.uniform(self.angle[0], self.angle[1]), direction

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        angle, direction = params
        if direction == "horizontal":
            data.img = tvF.affine(data.img, angle=0, translate=(0, 0), scale=1, shear=(angle, 0))
        else:
            data.img = tvF.affine(data.img, angle=0, translate=(0, 0), scale=1, shear=(0, angle))
        return data


@TRANSFORM_REGISTRY.register("random_shear_x")
class RandomShearX(RandomShear):
    def __init__(self, angle=(-17, 17)):
        super(RandomShearX, self).__init__(angle, "horizontal")


@TRANSFORM_REGISTRY.register("random_shear_y")
class RandomShearY(RandomShear):
    def __init__(self, angle=(-17, 17)):
        super(RandomShearY, self).__init__(angle, "vertical")


@TRANSFORM_REGISTRY.register("random_translate")
class RandomTranslate(Transforms):
    def __init__(self, direction="random", factor=0.3):
        self.direction = direction
        self.factor = factor

    def _get_params(self, data: GenericData):
        if self.direction == "random":
            direction = np.random.choice(["horizontal", "vertical"])
        else:
            direction = self.direction

        factor = np.random.uniform(self.factor, 0 - self.factor)
        if direction == "horizontal":
            offset = factor * data.img.shape[2]
        else:
            offset = factor * data.img.shape[1]

        return offset, direction

    def _process_img(self, data: GenericData, params: any) -> GenericData:
        offset, direction = params
        if direction == "horizontal":
            data.img = tvF.affine(data.img, angle=0, translate=(offset, 0), scale=1, shear=0)
        else:
            data.img = tvF.affine(data.img, angle=0, translate=(0, offset), scale=1, shear=0)

        return data


@TRANSFORM_REGISTRY.register("random_translate_x")
class RandomTranslateX(RandomTranslate):
    def __init__(self, factor=0.3):
        super(RandomTranslateX, self).__init__(factor=factor, direction="horizontal")


@TRANSFORM_REGISTRY.register("random_translate_y")
class RandomTranslateY(RandomTranslate):
    def __init__(self, factor=0.3):
        super(RandomTranslateY, self).__init__(factor=factor, direction="vertical")


@TRANSFORM_REGISTRY.register("batch_apply")
class BatchApply(Transforms):
    def __init__(self, **transforms):
        self.transforms = parse_transform_config(transforms)

    def __call__(self, batch: Iterable[GenericData]) -> Iterable[GenericData]:
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

    def _get_params(self, batch: Iterable[GenericData]):
        image_sizes = [data.img.shape[-2:] for data in batch]
        max_size = torch.tensor(image_sizes).max(0).values
        max_size = torch.div((max_size + (self.stride - 1)), self.stride, rounding_mode="trunc") * self.stride

        return max_size

    def _process_img(self, batch: Iterable[GenericData], params: any) -> Iterable[GenericData]:
        for i, data in enumerate(batch):
            pad_h = params[0] - int(data.img.shape[-2])
            pad_w = params[1] - int(data.img.shape[-1])
            data.img = tvF.pad(data.img, padding=[0, 0, pad_w, pad_h], fill=self.fill)
            data.meta.pad_size = [0, 0, int(pad_w), int(pad_h)]
            batch[i] = data
        return batch


@TRANSFORM_REGISTRY.register("random_select")
class RandomSelect(Transforms):
    def __init__(self, num, transforms):
        self.num = num
        self.transforms = parse_transform_config(transforms)

    def _get_params(self, data: GenericData):
        idx = torch.randperm(len(self.transforms))[: self.num].sort()[0]
        return idx

    def __call__(self, data: GenericData) -> GenericData:
        idx = self._get_params(data)
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

    def __call__(self, data: GenericData) -> GenericData:
        return self.transforms(data)

    def __repr__(self):
        trans_strs = []
        for transform in self.transforms:
            trans_strs.append(str(transform))
        trans_strs = ", ".join(trans_strs)
        return f"{self.__class__.__name__}({trans_strs})"


@TRANSFORM_REGISTRY.register("covert_color_channel")
class ConvertColorChannel(Transforms):
    def __init__(self, mode="BGR2RGB"):
        self.mode = mode

    def __call__(self, data: GenericData) -> GenericData:
        if self.mode == "BGR2RGB" or self.mode == "RGB2BGR":
            data.img = data.img[[2, 1, 0]]
        else:
            raise NotImplementedError("Unsupported mode: {}".format(self.mode))
        return data

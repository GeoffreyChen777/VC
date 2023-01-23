from typing import Any, Callable, Tuple
import torch
import numpy as np
from torchvision.ops.boxes import clip_boxes_to_image
import torchvision.transforms.functional as tvf
import torchvision.transforms as tv

from gvcore.utils.structure import GenericData


def resize(data: GenericData, min_size: Tuple[int], max_size: int, mode: str = "choice") -> GenericData:
    h, w = data.meta.cur_size
    if mode == "choice":
        size = np.random.choice(min_size)
    elif mode == "range":
        size = np.random.randint(min(min_size), max(min_size) + 1)
    else:
        raise ValueError("Unknown mode {}".format(mode))

    scale = size * 1.0 / min(h, w)
    if h < w:
        new_h, new_w = size, scale * w
    else:
        new_h, new_w = scale * h, size
    if max(new_h, new_w) > max_size:
        scale = max_size * 1.0 / max(new_h, new_w)
        new_h = new_h * scale
        new_w = new_w * scale
    new_w = int(new_w + 0.5)
    new_h = int(new_h + 0.5)

    data.img = tvf.resize(data.img, size=[new_h, new_w])
    resized_size = list(data.img.shape[1:])
    data.meta.cur_size = resized_size

    ratios = [s / s_orig for s, s_orig in zip(resized_size, (h, w))]
    ratio_height, ratio_width = ratios
    ratio = torch.tensor([[ratio_width, ratio_height, ratio_width, ratio_height]], device=data.label.device)
    if data.has("label"):
        data.label[:, :4] *= ratio
    return data


def normalise(data: GenericData, mean: Tuple[int], std: Tuple[int], scale: bool = False) -> GenericData:
    data.img = data.img[[2, 1, 0], :, :].float()
    if scale:
        data.img = data.img / 255.0
    if data.img.shape[0] == 1:
        data.img = data.img.repeat(3, 1, 1)
    data.img = tvf.normalize(data.img, mean, std)
    return data


def denormalise(data: GenericData, mean: Tuple[int], std: Tuple[int], scale: bool = False) -> GenericData:
    data.img = tvf.normalize(data.img, mean=[0.0, 0.0, 0.0], std=[1 / x for x in std])
    data.img = tvf.normalize(data.img, mean=[-1 * x for x in mean], std=[1.0, 1.0, 1.0])
    data.img = data.img[[2, 1, 0], :, :]
    if scale:
        data.img *= 255.0
    return data


def random_apply(data: Any, transform: Callable, p: float, *args, **kwargs) -> GenericData:
    if torch.rand(1) <= p:
        data = transform(data, *args, **kwargs)
    return data


def horizontal_flip(data: GenericData) -> GenericData:
    data.img = tvf.hflip(data.img)
    h, w = data.meta.cur_size
    if data.has("label"):
        data.label[:, [0, 2]] = w - data.label[:, [2, 0]]
    data.meta.flip = True
    return data


def gaussian_blur(data: GenericData, sigma: Tuple[float]) -> GenericData:
    kx = int(0.1 * data.meta.cur_size[1]) // 2 * 2 + 1
    ky = int(0.1 * data.meta.cur_size[0]) // 2 * 2 + 1
    data.img = tvf.gaussian_blur(data.img, [kx, ky], sigma)
    return data


def transform_label_by_size(
    label: torch.Tensor, from_size: Tuple[int], to_size: Tuple[int], flip=False
) -> torch.Tensor:
    label = label.clone()

    scale_x, scale_y = (
        1.0 * to_size[1] / from_size[1],
        1.0 * to_size[0] / from_size[0],
    )
    label[:, [0, 2]] *= scale_x
    label[:, [1, 3]] *= scale_y
    label[:, :4] = clip_boxes_to_image(label[:, :4], to_size)

    if flip:
        h, w = to_size
        label[:, [0, 2]] = w - label[:, [2, 0]]

    return label

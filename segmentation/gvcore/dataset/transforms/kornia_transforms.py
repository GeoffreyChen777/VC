from typing import Callable, Optional, Union
import numpy as np
import random
import torch
import kornia.augmentation as K
import kornia.enhance as KE
import kornia.geometry as KG


from gvcore.utils.structure import GenericData
from gvcore.dataset.transforms.transforms import TRANSFORM_REGISTRY


class KTransforms:
    def __init__(self, transform: K.AugmentationBase2D) -> None:
        self.transform = transform

    def _get_params(self, data: GenericData):
        if isinstance(self.transform, K.AugmentationBase2D):
            return self.transform._params
        else:
            return None

    def _process_img(self, data: GenericData) -> GenericData:
        data.img = self.transform(data.img)
        return data

    def _process_label(self, data: GenericData, params: any) -> GenericData:
        return data

    def __call__(self, data: GenericData) -> GenericData:
        data = self._process_img(data)
        params = self._get_params(data)
        data = self._process_label(data, params)
        return data

    def __repr__(self):
        params = []
        for k, v in self.__dict__.items():
            params.append(f"{k}: {v}")
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"


@TRANSFORM_REGISTRY.register("k_random_horizontal_flip")
class KRandomHorizontalFlip(KTransforms):
    def __init__(self, p=0.5, same_on_batch=False):
        super(KRandomHorizontalFlip, self).__init__(
            K.RandomHorizontalFlip(p=p, same_on_batch=same_on_batch, keepdim=True)
        )


@TRANSFORM_REGISTRY.register("k_random_crop")
class KRandomCrop(KTransforms):
    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", same_on_batch=False):
        super(KRandomCrop, self).__init__(
            K.RandomCrop(
                size=tuple(size),
                padding=tuple(padding),
                pad_if_needed=pad_if_needed,
                fill=fill,
                padding_mode=padding_mode,
                align_corners=False,
                same_on_batch=same_on_batch,
                keepdim=True,
            )
        )


class KRandomAdjust(KTransforms):
    def __init__(
        self,
        adjust_func: Callable[..., torch.Tensor],
        magnitude_limit=0.9,
        magnitude_bias=0.05,
        random_max=10,
        factor_type="float",
    ):
        super(KRandomAdjust, self).__init__(self._adjust_img)

        self.adjust_func = adjust_func
        self.magnitude_limit = magnitude_limit
        self.random_max = random_max
        self.magnitude_bias = magnitude_bias
        self.factor_type = factor_type

    def _get_random_params(self, img: Optional[torch.Tensor] = None) -> Union[float, int, torch.Tensor]:
        random_factor = random.randint(1, self.random_max + 1) / self.random_max
        adjust_factor = float(self.magnitude_limit) * random_factor
        if self.factor_type == "int":
            adjust_factor = int(adjust_factor)
        adjust_factor += self.magnitude_bias

        return adjust_factor

    def _adjust_img(self, img: torch.Tensor) -> torch.Tensor:
        adjust_factor = self._get_random_params(img)
        img = self.adjust_func(img, adjust_factor)
        return img


@TRANSFORM_REGISTRY.register("k_random_brightness")
class KRandomBrightness(KRandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(KRandomBrightness, self).__init__(KE.adjust_brightness, magnitude_limit, magnitude_bias, random_max)


@TRANSFORM_REGISTRY.register("k_random_saturation")
class KRandomSaturation(KRandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(KRandomSaturation, self).__init__(KE.adjust_saturation, magnitude_limit, magnitude_bias, random_max)


@TRANSFORM_REGISTRY.register("k_random_contrast")
class KRandomContrast(KRandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(KRandomContrast, self).__init__(KE.adjust_contrast, magnitude_limit, magnitude_bias, random_max)


@TRANSFORM_REGISTRY.register("k_equalize")
class KEqualize(KTransforms):
    def __init__(self, same_on_batch=True):
        super(KEqualize, self).__init__(K.RandomEqualize(p=1.0, same_on_batch=same_on_batch, keepdim=True))


@TRANSFORM_REGISTRY.register("k_random_posterize")
class KRandomPosterize(KRandomAdjust):
    def __init__(self, magnitude_limit=4, magnitude_bias=4, random_max=10):
        super(KRandomPosterize, self).__init__(KE.posterize, magnitude_limit, magnitude_bias, random_max, "int")


@TRANSFORM_REGISTRY.register("k_random_rotate")
class KRandomRotate(KRandomAdjust):
    def __init__(self, angle=(-30, 30)):
        super(KRandomRotate, self).__init__(
            KG.rotate, magnitude_limit=angle[1] - angle[0], magnitude_bias=angle[0], factor_type=int
        )

    def _get_random_params(self, img: Optional[torch.Tensor] = None) -> Union[float, int, torch.Tensor]:
        adjust_factor = super()._get_random_params(img)
        return torch.tensor([adjust_factor], device=img.device)


@TRANSFORM_REGISTRY.register("k_random_sharpness")
class KRandomSharpness(KRandomAdjust):
    def __init__(self, magnitude_limit=0.9, magnitude_bias=0.05, random_max=10):
        super(KRandomSharpness, self).__init__(KE.sharpness, magnitude_limit, magnitude_bias, random_max)


@TRANSFORM_REGISTRY.register("k_random_shear")
class KRandomShear(KRandomAdjust):
    def __init__(self, shear=(-0.3, 0.3), direction="random"):
        super(KRandomShear, self).__init__(KG.shear, magnitude_limit=shear[1] - shear[0], magnitude_bias=shear[0])
        self.direction = direction

    def _get_random_params(self, img: Optional[torch.Tensor] = None) -> Union[float, int, torch.Tensor]:
        adjust_factor = super(KRandomShear, self)._get_random_params()

        if self.direction == "random":
            direction = np.random.choice(["horizontal", "vertical"])
        else:
            direction = self.direction

        if direction == "horizontal":
            return torch.tensor([[adjust_factor, 0.0]], device=img.device)
        else:
            return torch.tensor([[0.0, adjust_factor]], device=img.device)


@TRANSFORM_REGISTRY.register("k_random_shear_x")
class KRandomShearX(KRandomShear):
    def __init__(self, shear=(-0.3, 0.3)):
        super(KRandomShearX, self).__init__(shear, "horizontal")


@TRANSFORM_REGISTRY.register("k_random_shear_y")
class KRandomShearY(KRandomShear):
    def __init__(self, shear=(-0.3, 0.3)):
        super(KRandomShearY, self).__init__(shear, "vertical")


@TRANSFORM_REGISTRY.register("k_random_solarize")
class KRandomSolarize(KRandomAdjust):
    def __init__(self, magnitude_limit=1.0, magnitude_bias=0.0, random_max=10):
        super(KRandomSolarize, self).__init__(None, magnitude_limit, magnitude_bias, random_max)

    def _adjust_img(self, img: torch.Tensor) -> torch.Tensor:
        threshold = self._get_random_params()
        img[img < threshold] = 1 - img[img < threshold]
        return img


@TRANSFORM_REGISTRY.register("k_random_translate")
class KRandomTranslate(KRandomAdjust):
    def __init__(self, translate=(-0.3, 0.3), direction="random"):
        super(KRandomTranslate, self).__init__(
            KG.translate, magnitude_limit=translate[1] - translate[0], magnitude_bias=translate[0]
        )
        self.direction = direction

    def _get_random_params(self, img: Optional[torch.Tensor] = None) -> Union[float, int, torch.Tensor]:
        adjust_factor = super(KRandomTranslate, self)._get_random_params()

        H, W = img.shape[-2:]
        if self.direction == "random":
            direction = np.random.choice(["horizontal", "vertical"])
        else:
            direction = self.direction

        if direction == "horizontal":
            return torch.tensor([[adjust_factor * W, 0.0]], device=img.device)
        else:
            return torch.tensor([[0.0, adjust_factor * H]], device=img.device)


@TRANSFORM_REGISTRY.register("k_random_translate_x")
class KRandomTranslateX(KRandomTranslate):
    def __init__(self, translate=(-0.3, 0.3)):
        super(KRandomTranslateX, self).__init__(translate, "horizontal")


@TRANSFORM_REGISTRY.register("k_random_translate_y")
class KRandomTranslateY(KRandomTranslate):
    def __init__(self, translate=(-0.3, 0.3)):
        super(KRandomTranslateY, self).__init__(translate, "vertical")


@TRANSFORM_REGISTRY.register("k_random_erase")
class KRandomErase(KTransforms):
    def __init__(self, scale=(0.02, 0.2), ratio=(0.1, 6.0), p=0.5, same_on_batch=False):
        super(KRandomErase, self).__init__(
            K.RandomErasing(scale=tuple(scale), ratio=tuple(ratio), p=p, same_on_batch=same_on_batch, keepdim=True)
        )

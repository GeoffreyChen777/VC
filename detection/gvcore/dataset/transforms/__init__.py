from copy import deepcopy

from gvcore.dataset import TRANSFORM_REGISTRY


__all__ = ["parse_transform_config", "DataCopy", "Compose", "Transforms"]


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


class DataCopy:
    """Copy data."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        data = deepcopy(data)
        data = self.transforms(data)
        return data

    def __repr__(self):
        return "[Deep Copied] \n" + str(self.transforms)


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "{0}".format(t)
        format_string += "\n)"
        return format_string

    def __getitem__(self, item):
        return self.transforms[item]

    def __len__(self):
        return len(self.transforms)


class Transforms:
    def __repr__(self):
        params = []
        for k, v in self.__dict__.items():
            params.append(f"{k}: {v}")
        params = ", ".join(params)
        return f"{self.__class__.__name__}({params})"


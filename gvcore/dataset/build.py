import torch

from gvcore.dataset.transforms import parse_transform_config
from gvcore.dataset.dataloader import BasicDataloader
from gvcore.dataset.sampler import GroupedBatchSampler, InfiniteSampler, SequentialSampler
from gvcore.dataset import DATASET_REGISTRY


def build_dataloader(cfg, subset="train"):
    # 0. Load data cache or path
    data_list_cache, data_list_path = None, None
    if cfg.data[subset].cache is not None:
        data_list_cache = torch.load(cfg.data[subset].cache)
    elif cfg.data[subset].path is not None:
        data_list_path = cfg.data[subset].path
    else:
        raise ValueError("No data cache or data path specified.")

    datacfg = cfg.data[subset]

    # 1. Construct transforms
    img_transforms = parse_transform_config(datacfg.img_transforms)
    batch_transforms = parse_transform_config(datacfg.batch_transforms)

    # 2. Construct dataset.
    dataset = DATASET_REGISTRY[datacfg.dataset](data_list_cache, data_list_path, img_transforms, subset)

    # 3. Construct sampler
    if not datacfg.infinity_sampler:
        data_sampler = SequentialSampler(size=len(dataset))
    else:
        data_sampler = InfiniteSampler(size=len(dataset), seed=cfg.seed)

    if datacfg.grouped_sampler:
        batch_sampler = GroupedBatchSampler(data_sampler, group_ids=dataset.group, batch_size=datacfg.batch_size)
    else:
        batch_sampler = None

    # 4. Construct dataloader
    data_loader = BasicDataloader(
        dataset,
        batch_size=datacfg.batch_size,
        sampler=data_sampler,
        batch_sampler=batch_sampler,
        num_workers=datacfg.num_workers,
        pin_memory=True,
        prefetch=datacfg.prefetch,
        batch_transforms=batch_transforms,
        drop_last=(subset != "test"),
    )

    return data_loader

from gvcore.dataset.transforms import parse_transform_config
from gvcore.dataset.dataloader import BasicDataloader
from gvcore.dataset.sampler import GroupedBatchSampler, InfiniteSampler, SequentialSampler
from gvcore.dataset import DATASET_REGISTRY


def build_dataloader(cfg, seed=669):
    # 1. Construct transforms
    img_transforms = parse_transform_config(cfg.img_transforms)
    batch_transforms = parse_transform_config(cfg.batch_transforms)

    # 2. Construct dataset.
    dataset = DATASET_REGISTRY[cfg.dataset](cfg.root, subset=cfg.subset, img_transforms=img_transforms)

    # 3. Construct sampler
    if not cfg.infinity_sampler:
        data_sampler = SequentialSampler(size=len(dataset))
    else:
        data_sampler = InfiniteSampler(size=len(dataset), seed=seed, shuffle=cfg.shuffle)

    if cfg.grouped_sampler:
        batch_sampler = GroupedBatchSampler(data_sampler, group_ids=dataset.group, batch_size=cfg.batch_size)
    else:
        batch_sampler = None

    # 4. Construct dataloader
    data_loader = BasicDataloader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=data_sampler,
        batch_sampler=batch_sampler,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        prefetch=cfg.get("prefetch", False),
        batch_transforms=batch_transforms,
        drop_last=cfg.get("drop_last", True),
    )

    return data_loader

import torch
from torch.utils.data.dataloader import DataLoader
from collections import Iterator


class Prefetcher(Iterator):
    def __init__(self, loader):
        self._loader = loader
        self.loader = iter(self._loader)
        self.stream = torch.cuda.Stream()
        self.next_batch = next(self.loader)
        self.to_device(self.next_batch, "cuda")
        torch.cuda.synchronize()

    @staticmethod
    def to_device(batch, device):
        if isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].to(device)
        elif isinstance(batch, list):
            for i, item in enumerate(batch):
                batch[i] = item.to(device)
        else:
            batch = batch.to(device)
        return batch

    def preload(self):
        try:
            self.next_batch = next(self.loader)
        except StopIteration:
            self.loader = iter(self._loader)
            self.next_batch = next(self.loader)
        with torch.cuda.stream(self.stream):
            self.to_device(self.next_batch, "cuda")

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.next_batch
        self.preload()
        return batch


class BasicDataloader:
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=None,
        prefetch=False,
        batch_transforms=None,
    ):
        if sampler is not None or batch_sampler is not None:
            shuffle = False
        if batch_sampler is not None:
            sampler = None
            batch_size = 1
            drop_last = False
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=dataset.collate_fn if collate_fn is None else collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            worker_init_fn=worker_init_fn,
        )

        self.batch_transforms = batch_transforms

        self.prefetch = prefetch
        if prefetch:
            self.iter_loader = Prefetcher(self.dataloader)
        else:
            self.iter_loader = iter(self.dataloader)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def to_device(batch, device):
        if isinstance(batch, dict):
            for key in batch:
                batch[key] = batch[key].to(device)
        elif isinstance(batch, list):
            for i, item in enumerate(batch):
                batch[i] = item.to(device)
        else:
            batch = batch.to(device)
        return batch

    @torch.no_grad()
    def get_batch(self, device="cuda"):
        try:
            batch = next(self.iter_loader)
            self.to_device(batch, device)
            if self.batch_transforms is not None:
                batch = self.batch_transforms(batch)
        except StopIteration:
            batch = None
            self.iter_loader = iter(self.dataloader)
        return batch

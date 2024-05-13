import torch
import logging
from datasets.gqa import GQADataset, gqa_collate
import time


def build_datasets(args):
    logger = logging.getLogger("isubgvqa")
    logger.info("build datasets")
    start_time = time.time()
    dataset_train = GQADataset(split="train")
    logger.info(f"{(time.time() - start_time):.2f}s elapsed to load the train dataset")
    start_time = time.time()
    dataset_valid = GQADataset(split="valid")
    logger.info(f"{(time.time() - start_time):.2f}s elapsed to load the train dataset")
    collate_fn = gqa_collate

    logger.info("build dataset samplers")
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        sampler_valid = torch.utils.data.DistributedSampler(dataset_valid, shuffle=True)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_valid = torch.utils.data.RandomSampler(dataset_valid)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=False
    )
    batch_sampler_valid = torch.utils.data.BatchSampler(
        sampler_valid,
        args.batch_size * 4,
        drop_last=False,
    )

    logger.info("build data loaders")
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_sampler=batch_sampler_valid,
    )

    datasets = {"train": dataloader_train, "dev": dataloader_valid}
    return datasets

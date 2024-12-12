import torch
import logging
from .gqa import GQADataset, gqa_collate
import time


def build_datasets(
    args=None,
    ans2label_path="./ISubGVQA/meta_info/trainval_ans2label.json",
    label2ans_path="./ISubGVQA/meta_info/trainval_label2ans.json",
):
    """
    Build and return datasets for training, validation, and test development.
    Args:
        args (Namespace, optional): Arguments containing configuration for dataset building.
        ans2label_path (str, optional): Path to the JSON file mapping answers to labels. Defaults to "./ISubGVQA/meta_info/trainval_ans2label.json".
        label2ans_path (str, optional): Path to the JSON file mapping labels to answers. Defaults to "./ISubGVQA/meta_info/trainval_label2ans.json".
    Returns:
        dict: A dictionary containing DataLoader objects for 'train', 'dev', and 'testdev' datasets.
    """
    logger = logging.getLogger("isubgvqa")
    logger.info("build datasets")
    start_time = time.time()
    dataset_train = GQADataset(
        split="train", ans2label_path=ans2label_path, label2ans_path=label2ans_path
    )
    logger.info(f"{(time.time() - start_time):.2f}s elapsed to load the train dataset")
    start_time = time.time()
    dataset_valid = GQADataset(
        split="valid", ans2label_path=ans2label_path, label2ans_path=label2ans_path
    )
    logger.info(f"{(time.time() - start_time):.2f}s elapsed to load the valid dataset")
    start_time = time.time()
    dataset_testdev = GQADataset(
        split="testdev", ans2label_path=ans2label_path, label2ans_path=label2ans_path
    )
    logger.info(
        f"{(time.time() - start_time):.2f}s elapsed to load the testdev dataset"
    )

    collate_fn = gqa_collate

    logger.info("build dataset samplers")
    if args.distributed:
        sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)
        sampler_valid = torch.utils.data.DistributedSampler(dataset_valid, shuffle=True)
        sampler_testdev = torch.utils.data.DistributedSampler(
            dataset_testdev, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_valid = torch.utils.data.RandomSampler(dataset_valid)
        sampler_testdev = torch.utils.data.RandomSampler(dataset_testdev)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=False
    )
    batch_sampler_valid = torch.utils.data.BatchSampler(
        sampler_valid,
        args.batch_size * 4,
        drop_last=False,
    )
    batch_sampler_testdev = torch.utils.data.BatchSampler(
        sampler_testdev,
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
    dataloader_testdev = torch.utils.data.DataLoader(
        dataset_testdev,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        batch_sampler=batch_sampler_testdev,
    )

    datasets = {
        "train": dataloader_train,
        "dev": dataloader_valid,
        "testdev": dataloader_testdev,
    }
    return datasets

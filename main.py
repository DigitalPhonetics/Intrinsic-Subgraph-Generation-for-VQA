import argparse
import logging
import os
import pathlib
import warnings

import torch
import torch.distributed as dist
import torch_geometric
from ignite.handlers.param_scheduler import create_lr_scheduler_with_warmup
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import ExponentialLR

from ISubGVQA.datasets.build import build_datasets
from ISubGVQA.models.build import build_model
from ISubGVQA.training.train_loop import train
from ISubGVQA.utils.arg_parser import get_argparser
from ISubGVQA.utils.config import get_config
from ISubGVQA.utils.misc import is_main_process

torch._dynamo.config.cache_size_limit = 64

# https://arxiv.org/abs/2109.08203
torch.manual_seed(3407)


def main(args: argparse.Namespace) -> None:
    """
    Main function to run the training and evaluation pipeline.
    Args:
        args (Namespace): Arguments and hyperparameters for the training and evaluation process.
    Initializes:
        - Logging configuration if output directory is specified and process is main
        - Distributed training setup if specified
        - Configuration, datasets, samplers, and dataloaders
        - Model and wraps it with DistributedDataParallel if distributed training is specified
        - Optimizer, gradient scaler, and learning rate scheduler
    Optionally resumes from a checkpoint if specified.
    Defines:
        - Loss criteria for short answer and node classification tasks
    Calls the train function with the initialized components.
    Cleans up distributed training setup if specified.
    Returns:
        None
    """

    print(f"torch version: {torch.__version__}")
    print(f"torch_geometric version: {torch_geometric.__version__}")
    print(
        f"os.environ['TOKENIZERS_PARALLELISM']={os.environ['TOKENIZERS_PARALLELISM']}"
    )

    args.batch_size = args.batch_size * args.scale_factor
    args.lr = args.lr * args.scale_factor

    print(f"scaled batch size of {args.batch_size}")
    print(f"scaled learning rate of {args.lr}")

    if args.output_dir and is_main_process():
        logging.basicConfig(
            filename=os.path.join(args.output_dir, args.log_name),
            filemode="w",
            format="%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s",
            level=logging.INFO,
        )
    warnings.filterwarnings("ignore")

    if is_main_process():
        logging.info(str(args))

    if args.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

    cfg = get_config(args=args)

    # init datasets, samplers, and dataloaders
    dataloaders = build_datasets(args=args)

    # init model
    model = build_model(args=args, cfg=cfg)

    if args.distributed:
        model.to(local_rank)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
        dist.barrier()
        torch.cuda.synchronize()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    # optimizer = torch.optim.AdamW(
    #     params=model.parameters(),
    #     lr=args.lr,
    #     weight_decay=1e-04,
    #     # samsgrad=True,
    # )

    optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=args.lr,
        # weight_decay=1e-04,
        # samsgrad=True,
    )

    gradscaler = GradScaler()

    torch_lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)

    lr_scheduler = create_lr_scheduler_with_warmup(
        torch_lr_scheduler,
        warmup_start_value=1e-6,
        warmup_end_value=args.lr,
        warmup_duration=10,
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint["model"])
            args = checkpoint["args"]
            if not args.evaluate:
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if "lr_scheduler" in checkpoint:
                    lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
                if "epoch" in checkpoint:
                    args.start_epoch = checkpoint["epoch"] + 1
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    criterion = {
        "short_answer": torch.nn.CrossEntropyLoss().to(device=args.device),
        "node_class": torch.nn.CrossEntropyLoss().to(device=args.device),
    }

    train(
        args=args,
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        gradscaler=gradscaler,
        lr_scheduler=lr_scheduler,
        save_model=True,
    )
    if args.distributed:
        cleanup()
    return


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Intrinsic Subgraph Generation for Graph based VQA", parents=[get_argparser()]
    )
    args = parser.parse_args()
    # if "LOCAL_RANK" not in os.environ:
    #     os.environ["LOCAL_RANK"] = str(args.local_rank)
    print(f"Current Working Directory: {os.getcwd()}")
    print(args)
    if args.output_dir:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

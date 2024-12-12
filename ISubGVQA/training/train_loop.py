import pathlib
import sys
import argparse

import torch
import torch.distributed as dist

from ..utils.misc import save_on_master

from .train_epoch import train_epoch
from .val_epoch import validate_epoch


def train(
    args: argparse.Namespace,
    model: torch.nn.Module,
    dataloaders: dict,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    gradscaler: torch.cuda.amp.GradScaler,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    save_model=False,
) -> float:
    """
    Train the model for a specified number of epochs.
    Args:
        args (Namespace): Arguments containing training configurations.
        model (torch.nn.Module): The model to be trained.
        dataloaders (dict): Dictionary containing 'train' and 'dev' dataloaders.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        gradscaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        sampler_train (torch.utils.data.Sampler): Sampler for the training data.
        save_model (bool, optional): Flag to save the model checkpoints. Defaults to False.
    Returns:
        float: The highest validation accuracy achieved during training.
    """

    if args.evaluate or args.pre_eval:
        validate_epoch(
            val_loader=dataloaders.get("dev"),
            model=model,
            criterion=criterion,
            args=args,
            epoch=-1,
        )
        if args.evaluate:
            return
    lr_scheduler(None)
    top_accuracy = 0
    loss_lowest = sys.maxsize
    for epoch in range(args.start_epoch, args.epochs):
        print(f'Learning rate of {optimizer.param_groups[0]["lr"]} in epoch {epoch}')
        if args.distributed:
            dist.barrier()
            torch.cuda.synchronize()
        train_epoch(
            dataloaders.get("train"),
            model,
            criterion,
            optimizer,
            epoch,
            args,
            gradscaler,
            grad_clipping=True,
        )
        # evaluate on validation set
        if args.distributed:
            dist.barrier()
            torch.cuda.synchronize()
        short_answer_acc, loss_val = validate_epoch(
            val_loader=dataloaders.get("dev"),
            model=model,
            criterion=criterion,
            args=args,
            epoch=epoch,
        )

        if args.distributed:
            dist.barrier()
            torch.cuda.synchronize()

        if loss_val < loss_lowest:
            loss_lowest = loss_val
            checkpoint_path = f"{args.output_dir}/checkpoint_lowest_val_loss.pth"
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )

        if short_answer_acc > top_accuracy:
            top_accuracy = short_answer_acc
            checkpoint_path = f"{args.output_dir}/checkpoint_top_res.pth"
            save_on_master(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                },
                checkpoint_path,
            )

        print(f"Top validation accuracy so far was {top_accuracy}")
        lr_scheduler(None)

        if save_model:
            output_dir = pathlib.Path(args.output_dir)
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            if (epoch + 1) % 50 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                save_on_master(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

    return top_accuracy

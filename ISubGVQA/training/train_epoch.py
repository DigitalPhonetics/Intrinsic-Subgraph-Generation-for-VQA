import copy
import logging
import math
import time

import torch
from utils.avg_meter import AverageMeter
from utils.progress_meter import ProgressMeter
from torch.cuda.amp import autocast
from utils.accuracies import accuracy
from torch.nn.parallel import DistributedDataParallel


def train_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    args,
    gradscaler,
    grad_clipping=True,
):
    progress_metrics = []
    batch_time = AverageMeter("Time", ":4.2f")
    progress_metrics.append(batch_time)
    data_time = AverageMeter("Data", ":4.2f")
    progress_metrics.append(data_time)
    losses = AverageMeter("Loss", ":.2e")
    progress_metrics.append(losses)
    ans_short = AverageMeter("Acc@Short-MGat", ":4.2f")
    progress_metrics.append(ans_short)

    if grad_clipping:
        model_params = model.parameters()

    progress = ProgressMeter(
        len(train_loader), progress_metrics, prefix="Epoch: [{}]".format(epoch)
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (data_batch) in enumerate(train_loader):
        if isinstance(model, torch.nn.DataParallel) or isinstance(
            model, DistributedDataParallel
        ):
            model.module.n_train_steps += args.batch_size
            train_steps = model.module.n_train_steps
        else:
            model.n_train_steps += args.batch_size
            train_steps = model.n_train_steps
        data_time.update(time.time() - end)
        (
            _,
            scene_graphs,
            questions,
            qsts_att_mask,
            short_answer_label,
            _,
            _,
        ) = data_batch  # programs
        # sdel questionID
        short_answer_label, scene_graphs = [
            datum.to(device="cuda", non_blocking=True)
            for datum in [short_answer_label, scene_graphs]
        ]

        if isinstance(questions, torch.Tensor):
            questions = questions.to(device="cuda")
            qsts_att_mask = qsts_att_mask.to(device="cuda")
            this_batch_size = questions.size(0)
        else:
            questions["input_ids"] = questions["input_ids"].to(device="cuda")
            questions["attention_mask"] = questions["attention_mask"].to(device="cuda")
            questions["token_type_ids"] = questions["token_type_ids"].to(device="cuda")

            this_batch_size = questions["input_ids"].size(1)

        output = model(
            node_embeddings=scene_graphs.x,
            edge_index=scene_graphs.edge_index,
            edge_embeddings=scene_graphs.edge_attr,
            batch=scene_graphs.batch,
            questions=questions,
            qsts_att_mask=qsts_att_mask,
            return_masks=True,
            scene_graphs=scene_graphs,
        )
        if isinstance(output, tuple):
            (
                short_answer_logits,
                imle_mask,
                gate,
                node_logits_layers,
                hidden_layers_short_answer_logits,
            ) = output
        else:
            short_answer_logits = output

        with torch.no_grad():
            this_short_answer_acc1 = accuracy(
                short_answer_logits, short_answer_label, topk=(1,)
            )
            ans_short.update(this_short_answer_acc1[0].item(), this_batch_size)

        loss = criterion["short_answer"](short_answer_logits, short_answer_label)

        optimizer.zero_grad()
        if gradscaler is not None:
            gradscaler.scale(loss).backward()
            gradscaler.unscale_(optimizer)
            if grad_clipping:
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=2.0)
            gradscaler.step(optimizer)
            gradscaler.update()
        else:
            loss.backward()
            optimizer.step()

        if not math.isnan(loss.item()):
            losses.update(loss.item(), this_batch_size)
        else:
            logging.info(f"loss is {loss.item()}")

        ##################################
        # measure elapsed time
        ##################################
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i == len(train_loader) - 1:
            progress.display(i)

    ##################################
    # Give final score
    ##################################
    progress.display(batch=len(train_loader))

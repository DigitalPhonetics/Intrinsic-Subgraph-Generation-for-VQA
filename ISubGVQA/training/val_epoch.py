import time

import torch
from torch.nn.parallel import DistributedDataParallel

from utils.accuracies import accuracy

from utils.avg_meter import AverageMeter
from utils.progress_meter import ProgressMeter


def validate_epoch(val_loader, model, criterion, args, epoch):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.2e")
    mgat_losses = AverageMeter("MGat-Loss", ":.4e")
    ans_short = AverageMeter("Acc@Short-MGat", ":4.2f")

    progress_metrics = [
        batch_time,
        losses,
        mgat_losses,
        ans_short,
    ]

    progress = ProgressMeter(len(val_loader), progress_metrics, prefix="Test: ")

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (data_batch) in enumerate(val_loader):
            if isinstance(model, torch.nn.DataParallel) or isinstance(
                model, DistributedDataParallel
            ):
                model.module.n_valid_steps += args.batch_size
                valid_steps = model.module.n_valid_steps
            else:
                model.n_valid_steps += args.batch_size
                valid_steps = model.n_valid_steps

            (
                _,
                scene_graphs,
                questions,
                qsts_att_mask,
                short_answer_label,
                _,
                _,
            ) = data_batch

            short_answer_label, scene_graphs = [
                datum.to(device="cuda", non_blocking=True)
                for datum in [short_answer_label, scene_graphs]
            ]
            if isinstance(questions, torch.Tensor):
                questions = questions.to(device="cuda")
                this_batch_size = questions.size(0)
                qsts_att_mask = qsts_att_mask.to(device="cuda")
            else:
                questions["input_ids"] = questions["input_ids"].to(device="cuda")
                questions["attention_mask"] = questions["attention_mask"].to(
                    device="cuda"
                )
                questions["token_type_ids"] = questions["token_type_ids"].to(
                    device="cuda"
                )

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

            this_short_answer_acc1 = accuracy(
                short_answer_logits.detach(), short_answer_label, topk=(1,)
            )
            loss = criterion["short_answer"](short_answer_logits, short_answer_label)
            losses.update(loss.item(), this_batch_size)

            ans_short.update(this_short_answer_acc1[0].item(), this_batch_size)
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0 or i == len(val_loader) - 1:
                progress.display(i)

    progress.display(batch=len(val_loader))

    return (ans_short.avg, losses.avg)

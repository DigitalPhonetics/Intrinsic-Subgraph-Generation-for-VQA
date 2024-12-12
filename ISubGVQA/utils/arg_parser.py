import argparse


def get_argparser():
    parser = argparse.ArgumentParser("ISubGVQA", add_help=False)

    #
    parser.add_argument("--config", default="./ISubGVQA/configs/config_default.json")
    # parser.add_argument("--data", metavar="PATH", default="./", help="path to dataset")
    # parser.add_argument(
    #     "--save-dir", metavar="PATH", default="./", help="path to dataset"
    # )
    parser.add_argument("--mgat_layers", default=4, type=int, metavar="N")
    parser.add_argument("--log-name", default="gtsg.log", type=str, metavar="PATH")
    parser.add_argument("--num_workers", default=4, type=int, metavar="N")
    parser.add_argument("--epochs", default=100, type=int, metavar="N")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N")
    parser.add_argument("--nb_samples", default=1, type=int, metavar="N")
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--beta", default=10.0, type=float)
    parser.add_argument("--tau", default=1.0, type=float)

    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=5e-5,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument("--lr_drop", default=30, type=int)
    parser.add_argument("--scale_factor", default=1, type=int)
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=50,
        type=int,
        metavar="N",
        help="print frequency (default: 50)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--evaluate_sets",
        default=["val_unbiased", "testdev"],
        nargs="+",
        help="Data sets/splits to perform evaluation, e.g. "
        "val_unbiased, testdev etc. Multiple sets/splits "
        "are supported and need to be separated by space",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--output_dir",
        default="./outputdir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--work_dir",
        default="./outputdir",
        help="path where to save, empty for no saving",
    )

    parser.add_argument("--gnn_gating", type=int, default=1)
    parser.add_argument("--use_instruction", type=int, default=1)
    parser.add_argument("--use_masking", type=int, default=1)
    parser.add_argument("--use_mgat", type=int, default=0)
    parser.add_argument(
        "--mgat_masks", nargs="+", type=float, default=[1.0, 1.0, 1.0, 0.15]
    )
    parser.add_argument("--use_topk", type=int, default=True)
    parser.add_argument("--interpretable_mode", type=int, default=False)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--concat_instr", type=int, default=0)
    parser.add_argument("--embed_cat", type=int, default=0)
    parser.add_argument("--use_subgat", action="store_true", default=False)
    parser.add_argument("--node_classification", action="store_true", default=False)
    parser.add_argument("--bi_qa", action="store_true", default=False)
    parser.add_argument("--general_hidden_dim", type=int, default=300)
    parser.add_argument("--use_all_instrs", action="store_true", default=False)
    parser.add_argument("--use_global_mask", action="store_true", default=False)
    parser.add_argument("--mask_regularization", action="store_true", default=False)
    parser.add_argument("--pre_eval", action="store_true", default=False)
    parser.add_argument("--distributed", action="store_true", default=False)
    parser.add_argument("--text_sampling", action="store_true", default=False)

    parser.add_argument("--sampler_type", type=str)
    parser.add_argument("--sample_k", type=int)

    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="ids of gpus to use " "(only applicable to non-distributed training)",
    )

    return parser

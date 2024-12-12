from .isubgvqa import ISubGVQA


def build_model(args, cfg):
    """
    Build and initialize the ISubGVQA model with the given arguments and configuration.
    Args:
        args (Namespace): A namespace object containing the arguments for model initialization.
        cfg (dict): A dictionary containing the configuration settings.
    Returns:
        ISubGVQA: An instance of the ISubGVQA model initialized with the specified parameters.
    """

    model = ISubGVQA(
        args,
        use_imle=True,
        use_masking=args.use_masking,
        use_instruction=args.use_instruction,
        use_mgat=args.use_mgat,
        mgat_masks=args.mgat_masks,
        use_topk=args.use_topk,
        interpretable_mode=args.interpretable_mode,
        concat_instr=args.concat_instr,
        embed_cat=args.embed_cat,
    )
    model.to(device=args.device)
    return model

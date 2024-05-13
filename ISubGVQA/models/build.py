from .isubgvqa import ISubGVQA


def build_model(args, cfg):

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

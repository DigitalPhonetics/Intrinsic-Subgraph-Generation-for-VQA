import torch
import torch_geometric

from ..utils.scatter_scaled_dot_product import scatter_scaled_dot_product_attention
from .mgat_v2_conv import MaskingGATv2Conv


class MGAT(torch.nn.Module):
    def __init__(
        self,
        channels,
        num_ins,
        dropout=0.0,
        heads=4,
        use_instr=False,
        masking_thresholds=None,
        use_topk: bool = False,
        interpretable_mode: bool = True,
        concat_instr: bool = False,
        use_all_instrs: bool = False,
        use_global_mask: bool = False,
        node_classification: bool = False,
        sampler_type: str = None,
        sample_k: int = None,
        nb_samples: int = 1,
        alpha=1.0,
        beta=10.0,
        tau=1.0,
    ):
        super(MGAT, self).__init__()
        self.masking_thresholds = masking_thresholds
        self.use_global_mask = use_global_mask
        self.node_classification = node_classification

        self.heads = heads
        self.use_instr = use_instr
        self.use_topk = use_topk
        self.interpretable_mode = interpretable_mode
        self.use_all_instrs = use_all_instrs

        if concat_instr:
            self.in_channels = channels * 2
        else:
            self.in_channels = channels

        if use_instr:
            self.use_instr_list = [True, True, True, True]
            self.in_channels_list = [
                self.in_channels,
                self.in_channels,
                self.in_channels,
                self.in_channels,
            ]

        self.convs = torch.nn.ModuleList(
            [
                MaskingGATv2Conv(
                    in_channels=self.in_channels_list[i],
                    out_channels=channels,
                    heads=heads,
                    edge_dim=channels,
                    masking_threshold=self.masking_thresholds[i],
                    add_self_loops=False,
                    use_instr=self.use_instr_list[i],
                    use_topk=use_topk,
                    concat_instr=concat_instr,
                    use_all_instrs=use_all_instrs,
                    sampler_type=sampler_type,
                    sample_k=sample_k,
                    nb_samples=nb_samples,
                    alpha=alpha,
                    beta=beta,
                    tau=tau,
                )
                for i in range(num_ins)
            ]
        )

        self.x_proj = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(heads * channels, channels * int(heads / 2)),
                    torch.nn.GELU(),
                    torch.nn.Linear(channels * int(heads / 2), channels),
                    torch.nn.GELU(),
                )
                for i in range(num_ins)
            ]
        )

        # for the last output, no batch norm
        # self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(channels) for _ in range(num_ins-1)])
        self.bns = torch.nn.ModuleList(
            [torch_geometric.nn.norm.GraphNorm(channels) for _ in range(num_ins)]
        )
        self.dropout = dropout

        self.node_logits = torch.nn.Sequential(
            torch.nn.Linear(channels, 512),
            torch.nn.GELU(),
            torch.nn.Linear(512, 2577),
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(
        self,
        x,
        edge_index,
        instr_vectors,
        global_language_feats,
        edge_attr,
        batch,
        return_masks=False,
        explainer=False,
        explainer_stage=False,
        expl_bypass_x=False,
    ):
        h = x

        node_logits_layers = []
        hidden_states = []

        node_masks = []
        edge_attns = []

        num_conv_layers = len(self.convs)

        # construct initial global mask across layers
        if self.use_global_mask:
            global_mask = torch.ones((h.size(0), 1), device=h.device, dtype=h.dtype)

        for i in range(num_conv_layers):
            ins = instr_vectors[i]

            if explainer:
                h = expl_bypass_x if (explainer_stage - 1) == i else h

            # perform vector message passing
            conv_res, mask, edge_att = self.convs[i](
                x=h,
                edge_index=edge_index,
                edge_attr=edge_attr,
                instruction=ins,
                batch=batch,
                return_masks=return_masks,
                return_attention_weights=True,
                imle_att=global_language_feats,
                all_instrs=instr_vectors,
            )
            # project heads into a joint dimension
            conv_res = self.x_proj[i](conv_res)

            node_masks.append(mask)
            edge_attns.append(edge_att)

            if self.use_global_mask:
                global_mask = mask * global_mask

            # if i == 0 and self.interpretable_mode:
            #     h_state = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
            #     h = conv_res + h_state
            # else:
            conv_res = scatter_scaled_dot_product_attention(
                ins, conv_res, conv_res, batch
            )
            conv_res = self.bns[i](conv_res, batch=batch)
            h = conv_res + h

            if self.use_global_mask:
                h = global_mask * h
            elif self.interpretable_mode and mask != None:
                h = mask * h

        return (
            h,
            mask,
            node_logits_layers,
            hidden_states,
        )

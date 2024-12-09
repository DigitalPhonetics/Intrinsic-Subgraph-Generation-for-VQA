from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag

from ..sampling.node_edge_masks import NodeMaskToEdgeMask
from .masking import MaskingModel


class MaskingGATv2Conv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, torch.Tensor, str] = "mean",
        bias: bool = True,
        share_weights: bool = False,
        masking_threshold: Optional[int] = None,
        use_instr: bool = False,
        use_topk: bool = False,
        concat_instr: bool = False,
        use_all_instrs: bool = False,
        sampler_type: str = None,
        sample_k: int = None,
        nb_samples: int = 1,
        alpha=1.0,
        beta=10.0,
        tau=1.0,
        **kwargs,
    ):
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value
        self.share_weights = share_weights
        self.use_instr = use_instr
        self.concat_instr = concat_instr
        self.use_all_instrs = use_all_instrs

        if isinstance(in_channels, int):
            self.lin_l = Linear(
                in_channels,
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels,
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )
        else:
            self.lin_l = Linear(
                in_channels[0],
                heads * out_channels,
                bias=bias,
                weight_initializer="glorot",
            )
            if share_weights:
                self.lin_r = self.lin_l
            else:
                self.lin_r = Linear(
                    in_channels[1],
                    heads * out_channels,
                    bias=bias,
                    weight_initializer="glorot",
                )

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        if edge_dim is not None:
            self.lin_edge = Linear(
                edge_dim, heads * out_channels, bias=False, weight_initializer="glorot"
            )
        else:
            self.lin_edge = None

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self._alpha = None

        self.mask = MaskingModel(
            in_channels,
            out_channels,
            masking_threshold,
            use_topk=use_topk,
            sampler_type=sampler_type,
            sample_k=sample_k,
            nb_samples=nb_samples,
            alpha=alpha,
            beta=beta,
            tau=tau,
        )
        self.masking = NodeMaskToEdgeMask.apply

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()
        if self.lin_edge is not None:
            self.lin_edge.reset_parameters()
        glorot(self.att)
        zeros(self.bias)

    def forward(
        self,
        x: Union[torch.Tensor, PairTensor],
        edge_index: Adj,
        batch: Union[torch.Tensor, PairTensor],
        edge_attr: OptTensor = None,
        instruction: OptTensor = None,
        imle_att: OptTensor = None,
        return_attention_weights: bool = None,
        return_masks: bool = None,
        all_instrs=None,
    ):
        H, C = self.heads, self.out_channels

        if self.use_instr:
            if self.concat_instr:
                x = torch.cat((x, instruction[batch]), dim=1)
            else:
                x = x * instruction[batch]
                x = F.gelu(x)

        mask = None
        mask_edge_weight = None
        if self.mask.masking_threshold != 1.0:
            if self.use_all_instrs:
                mask = self.mask(x, all_instrs, batch, edge_index, use_all_instrs=True)
                mask = mask.unsqueeze(-1)
            else:
                mask = self.mask(
                    x, imle_att[batch], batch, edge_index, use_all_instrs=False
                )
            mask_edge_weight = self.masking(
                mask, edge_index, torch.tensor(mask.shape[0])
            )

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
                edge_index, edge_attr = add_self_loops(
                    edge_index,
                    edge_attr,
                    fill_value=self.fill_value,
                    num_nodes=num_nodes,
                )
            elif isinstance(edge_index, SparseTensor):
                if self.edge_dim is None:
                    edge_index = set_diag(edge_index)
                else:
                    raise NotImplementedError(
                        "The usage of 'edge_attr' and 'add_self_loops' "
                        "simultaneously is currently not yet supported for "
                        "'edge_index' in a 'SparseTensor' form"
                    )

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(
            edge_index,
            x=(x_l, x_r),
            edge_attr=edge_attr,
            edge_mask=mask_edge_weight,
            size=None,
        )

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, mask, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout="coo")
        else:
            return out, mask

    def message(
        self,
        x_j: Tensor,
        x_i: Tensor,
        edge_attr: OptTensor,
        edge_mask,
        index: Tensor,
        ptr: OptTensor,
        size_i: Optional[int],
    ) -> Tensor:
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x += edge_attr

        if edge_mask is not None:
            x = x * edge_mask.unsqueeze(-1)

        x = F.leaky_relu(x, self.negative_slope)

        if edge_mask is not None:
            x = x * edge_mask.unsqueeze(-1)

        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)

        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        if edge_mask is None:
            return x_j * alpha.unsqueeze(-1)
        return x_j * (alpha * edge_mask).unsqueeze(-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_channels}, "
            f"{self.out_channels}, heads={self.heads})"
        )

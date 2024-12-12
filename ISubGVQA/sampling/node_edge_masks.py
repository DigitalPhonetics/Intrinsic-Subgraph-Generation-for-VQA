import torch
from torch_scatter import scatter


class NodeMaskToEdgeMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mask, *args):
        edge_index, n_nodes = args
        ctx.save_for_backward(mask, edge_index, n_nodes)
        return (mask[edge_index[0]] * mask[edge_index[1]]).to(torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        mask, edge_index, n_nodes = ctx.saved_tensors
        grad_input = grad_output.clone()
        final_grad = scatter(
            grad_output, edge_index[1], dim=0, reduce="sum", dim_size=n_nodes
        )
        return final_grad, None, None

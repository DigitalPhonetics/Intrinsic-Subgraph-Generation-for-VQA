import torch
import torch_geometric
from torch_scatter import scatter_add


class GlobalAttention(torch.nn.Module):
    def __init__(self, num_node_features, num_out_features):
        super(GlobalAttention, self).__init__()
        channels = num_out_features
        self.gate_nn = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, 1),
        )
        self.node_nn = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, channels),
        )
        self.ques_nn = torch.nn.Sequential(
            torch.nn.Linear(channels, channels),
            torch.nn.GELU(),
            torch.nn.Linear(channels, channels),
        )

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.gate_nn)
        torch_geometric.nn.inits.reset(self.node_nn)
        torch_geometric.nn.inits.reset(self.ques_nn)

    def forward(self, x, u, batch, size=None, return_mask=False, node_mask=None):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        # gate = self.gate_nn(x).view(-1, 1)

        ##################################
        # Batch
        # shape: x - [ Num of Nodes, num_node_features] --> [ Num of Nodes, Feature Channels ]
        # shape: u - [ Batch Size, Feature Channels]
        # shape: u[batch] - [ Num of Nodes, Feature Channels]
        ##################################
        x = self.node_nn(x)  # if self.node_nn is not None else x
        # print("x", x.size(), "u", u.size(), "u[batch]", u[batch].size())
        if node_mask is not None:
            x = x * node_mask
        ##################################
        # torch.bmm
        # batch1 and batch2 must be 3D Tensors each containing the same number of matrices.
        # If batch1 is a b x n x m Tensor, batch2 is a b x m x p Tensor, out will be a b x n x p Tensor.
        ##################################

        # gate = self.gate_nn(self.ques_nn(u)[batch] * x)
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch.bmm(x.unsqueeze(1), self.ques_nn(u)[batch].unsqueeze(2)).squeeze(
            -1
        ) / torch.sqrt(torch.tensor(x.size(1)))
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch.cuda(), num_nodes=size)
        # gate = pyg_old_softmax(gate, batch.cuda(), num_nodes=size)

        out = scatter_add(gate * x, batch.cuda(), dim=0, dim_size=size)

        if return_mask:
            return out, gate
        return out

    def __repr__(self):
        return "{}(gate_nn={}, node_nn={}, ques_nn={})".format(
            self.__class__.__name__, self.gate_nn, self.node_nn, self.ques_nn
        )

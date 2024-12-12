import torch
import torch_geometric
from torch_scatter import scatter_add


class GlobalAttention(torch.nn.Module):
    """
    GlobalAttention is a neural network module that applies attention mechanisms to graph data.
    Args:
        num_node_features (int): The number of input features per node.
        num_out_features (int): The number of output features.
    Methods:
        reset_parameters():
            Resets the parameters of the neural network layers.
        forward(x, u, batch, size=None, return_mask=False, node_mask=None):
            Forward pass of the GlobalAttention module.
            Args:
                x (Tensor): Node feature matrix with shape [num_nodes, num_node_features].
                u (Tensor): Global feature matrix with shape [batch_size, num_out_features].
                batch (Tensor): Batch vector which assigns each node to a specific example in the batch.
                size (int, optional): The number of examples in the batch. If None, it is inferred from the batch vector.
                return_mask (bool, optional): If True, returns the attention mask along with the output.
                node_mask (Tensor, optional): Mask to apply on the node features.
            Returns:
                Tensor: The output feature matrix with shape [batch_size, num_out_features].
                Tensor (optional): The attention mask with shape [num_nodes, 1] if return_mask is True.
        __repr__():
            Returns a string representation of the GlobalAttention module.
    """

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

        x = self.node_nn(x)
        if node_mask is not None:
            x = x * node_mask

        gate = torch.bmm(x.unsqueeze(1), self.ques_nn(u)[batch].unsqueeze(2)).squeeze(
            -1
        ) / torch.sqrt(torch.tensor(x.size(1)))
        assert gate.dim() == x.dim() and gate.size(0) == x.size(0)

        gate = torch_geometric.utils.softmax(gate, batch.cuda(), num_nodes=size)

        out = scatter_add(gate * x, batch.cuda(), dim=0, dim_size=size)

        if return_mask:
            return out, gate
        return out

    def __repr__(self):
        return "{}(gate_nn={}, node_nn={}, ques_nn={})".format(
            self.__class__.__name__, self.gate_nn, self.node_nn, self.ques_nn
        )

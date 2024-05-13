import torch
import torch_geometric
import torch.nn.functional as F
from utils.topk import topk


class MaskingModel(torch.nn.Module):
    def __init__(self, dim_nodes, dim_questions, masking_threshold=0.3, use_topk=False):
        super(MaskingModel, self).__init__()
        # self.gate_nn = Seq(Linear(channels, channels), GELU(), Linear(channels, 1))
        # self.node_nn = Seq(Linear(num_node_features, channels), GELU(), Linear(channels, channels))
        # self.ques_nn = Seq(Linear(channels, channels), GELU(), Linear(channels, channels))
        self.use_topk = use_topk
        self.masking_threshold = (
            int(masking_threshold) if masking_threshold > 1 else masking_threshold
        )

        self.dim_nodes = dim_nodes
        self.dim_questions = dim_questions

        self.gate_nn = torch.nn.Sequential(
            torch.nn.Linear(dim_questions, dim_questions),
            torch.nn.GELU(),
            torch.nn.Linear(dim_questions, 1),
        )
        self.node_nn = torch.nn.Sequential(
            torch.nn.Linear(dim_nodes, dim_questions), torch.nn.GELU()
        )
        self.ques_nn = torch.nn.Sequential(
            torch.nn.Linear(dim_questions, dim_questions), torch.nn.GELU()
        )

        if use_topk:
            self.gate_top = torch_geometric.nn.TopKPooling(in_channels=dim_questions)

        self.reset_parameters()

    def reset_parameters(self):
        torch_geometric.nn.inits.reset(self.gate_nn)
        torch_geometric.nn.inits.reset(self.node_nn)
        torch_geometric.nn.inits.reset(self.ques_nn)

    def forward(self, x, u, batch, edge_index, size=None, use_all_instrs=True):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        size = batch[-1].item() + 1 if size is None else size

        x = self.node_nn(x)

        # gate = self.gate_nn(self.ques_nn(u)[batch] * x)
        # assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
        if use_all_instrs:
            (
                node_embeddings_batch,
                batch_info_batch,
            ) = torch_geometric.utils.to_dense_batch(x, batch=batch)
            inst_vectors = u.transpose(0, 1)
            scores = torch.einsum("bmd,bnd->bmn", inst_vectors, node_embeddings_batch)
            attention = torch.softmax(scores, dim=2)
            gate = attention.sum(1)[batch_info_batch]
        else:
            gate = torch.bmm(
                x.unsqueeze(1), self.ques_nn(u)[batch].unsqueeze(2)
            ).squeeze(-1) / torch.sqrt(torch.tensor(x.size(1)))
            assert gate.dim() == x.dim() and gate.size(0) == x.size(0)
            gate = F.gelu(gate)
            # gate = F.sigmoid(gate)
        # gate = torch_geometric.utils.softmax(gate, batch.cuda(), num_nodes=size)
        if self.use_topk:
            gate = F.dropout(gate, p=0.2, training=self.training)
            gate_indices = topk(gate.squeeze(), self.masking_threshold, batch=batch)
            neg_indices = torch.tensor(
                [
                    i
                    for i, e in enumerate(range(batch.size(0)))
                    if i not in gate_indices
                ],
                device=x.device,
            )
            gate.index_fill_(0, gate_indices, 1)
            if neg_indices.nelement() != 0:
                gate.index_fill_(0, neg_indices, 0)
        else:
            gate = F.dropout(gate, p=0.2, training=self.training)
            gate = F.sigmoid(gate)
            gate = (gate > 0.5).to(dtype=gate.dtype)
        return gate

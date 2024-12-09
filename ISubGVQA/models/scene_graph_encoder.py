import torch
import torch_geometric
from torch_scatter import scatter_mean

from ..datasets.scene_graph import GQASceneGraphs


class SceneGraphEncoder(torch.nn.Module):
    def __init__(self, hidden_dim, dist=False):
        super(SceneGraphEncoder, self).__init__()
        self.scene_graphs_helper = GQASceneGraphs()

        self.sg_vocab = self.scene_graphs_helper.vocab_sg
        self.hidden_dim = hidden_dim
        self.dist = dist

        self.sg_emb_dim = 300
        sg_pad_idx = self.sg_vocab.vocab.itos_.index("<pad>")
        self.sg_vocab_embedding = torch.nn.Embedding(
            len(self.sg_vocab.vocab), self.sg_emb_dim, padding_idx=sg_pad_idx
        )
        self.sg_vocab_embedding.weight.data.copy_(self.scene_graphs_helper.vectors)
        ##################################
        # build scene graph encoding layer
        ##################################
        self.scene_graph_encoding_layer = get_gt_scene_graph_encoding_layer(
            num_node_features=self.sg_emb_dim,
            num_edge_features=self.sg_emb_dim,
            hidden_dim=hidden_dim,
        )

        # TODO: double check with inital code
        self.graph_layer_norm = torch_geometric.nn.norm.GraphNorm(self.sg_emb_dim)

        self.bbox_encoding = torch.nn.Sequential(
            torch.nn.SyncBatchNorm(4) if dist else torch.nn.BatchNorm1d(4),
            torch.nn.Linear(4, 16),
            torch.nn.GELU(),
            torch.nn.SyncBatchNorm(16) if dist else torch.nn.BatchNorm1d(16),
            torch.nn.Linear(16, 32),
            torch.nn.GELU(),
        )
        self.feat_reduc = torch.nn.Sequential(
            (
                torch.nn.SyncBatchNorm(self.sg_emb_dim + 32)
                if dist
                else torch.nn.BatchNorm1d(self.sg_emb_dim + 32)
            ),
            torch.nn.Linear(self.sg_emb_dim + 32, self.sg_emb_dim),
            torch.nn.GELU(),
        )

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        batch,
        explainer=False,
        explainer_stage=False,
        gt_scene_graphs=None,
    ):
        x_embed = (
            x if explainer and (explainer_stage == 0) else self.sg_vocab_embedding(x)
        )
        x_embed_sum = (
            x_embed
            if explainer and (explainer_stage == 0)
            else torch.sum(input=x_embed, dim=-2, keepdim=False)
        )

        x_bbox = self.bbox_encoding(gt_scene_graphs.x_bbox.to(dtype=x_embed.dtype))
        x_embed_sum = torch.cat((x_embed_sum, x_bbox), dim=1)
        x_embed_sum = self.feat_reduc(x_embed_sum)

        edge_attr_embed = self.sg_vocab_embedding(edge_attr)

        # edge_attr_embed[gt_scene_graphs.added_sym_edge, :, :] *= -1
        # edge_attr_embed[gt_scene_graphs.added_sym_edge, :] *= -1
        edge_attr_embed[gt_scene_graphs.added_sym_edge, :] *= -1

        # [ num_edges, MAX_EDGE_TOKEN_LEN, sg_emb_dim] -> [ num_edges, sg_emb_dim]
        # edge_attr_embed_sum   = torch.sum(input=edge_attr_embed, dim=-2, keepdim=True)
        edge_attr_embed_sum = edge_attr_embed
        del x_embed, edge_attr_embed

        ##################################
        # Call scene graph encoding layer
        ##################################

        x_encoded, edge_attr_encoded, _ = self.scene_graph_encoding_layer(
            x=x_embed_sum,
            edge_index=edge_index,
            edge_attr=edge_attr_embed_sum,
            u=None,
            batch=batch,
        )

        save_type = x_encoded.dtype
        x_encoded = x_encoded.type(torch.DoubleTensor).to(device=x.device)
        x_encoded = self.graph_layer_norm(x_encoded, batch.to(device=x.device))
        x_encoded = x_encoded.type(save_type).to(device=x.device)

        return x_encoded, edge_attr_encoded


def get_gt_scene_graph_encoding_layer(num_node_features, num_edge_features, hidden_dim):
    class EdgeModel(torch.nn.Module):
        def __init__(self, hidden_dim=300):
            super(EdgeModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * num_node_features + num_edge_features, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, src, dest, edge_attr, u, batch):
            out = torch.cat([src, dest, edge_attr], 1)
            return self.edge_mlp(out)

    class NodeModel(torch.nn.Module):
        def __init__(self, hidden_dim=300):
            super(NodeModel, self).__init__()
            self.hidden_dim = hidden_dim
            self.node_mlp_1 = torch.nn.Sequential(
                torch.nn.Linear(num_node_features + hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )
            self.node_mlp_2 = torch.nn.Sequential(
                torch.nn.Linear(num_node_features + hidden_dim, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
            )

        def forward(self, x, edge_index, edge_attr, u, batch):
            row, col = edge_index
            out = torch.cat([x[row], edge_attr], dim=1)
            out = self.node_mlp_1(out)
            out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
            out = torch.cat([x, out], dim=1)
            return self.node_mlp_2(out)

    op = torch_geometric.nn.MetaLayer(EdgeModel(hidden_dim), NodeModel(hidden_dim))
    return op

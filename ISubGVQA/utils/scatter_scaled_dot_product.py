from torch_scatter import scatter_softmax
import math
import torch


def scatter_scaled_dot_product_attention(query, key, value, batch):
    attention = scatter_softmax(
        torch.bmm(
            query[batch].unsqueeze(1), key.unsqueeze(1).transpose(-2, -1)
        ).squeeze()
        / math.sqrt(query.size(-1)),
        batch,
        dim=-1,
    )
    return attention.unsqueeze(1) * value

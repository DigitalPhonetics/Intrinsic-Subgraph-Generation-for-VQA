import torch
from .positional_encoding import PositionalEncoding
import math


class QuestionEncoder(torch.nn.Module):
    def __init__(
        self,
        text_vocab_embedding,
        text_emb_dim,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.5,
    ):
        super(QuestionEncoder, self).__init__()
        self.text_vocab_embedding = text_vocab_embedding
        self.model_type = "Transformer"
        self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = torch.nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp)
        )
        self.ninp = ninp

    def forward(self, src, mask):
        ##################################
        # forward model, expect [Len, Batch, Dim]
        ##################################
        src = self.text_vocab_embedding(src)
        # src = self.emb_proj(src) * math.sqrt(self.ninp)
        # src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src.permute(1, 0, 2), src_key_padding_mask=mask.float()
        )
        return output

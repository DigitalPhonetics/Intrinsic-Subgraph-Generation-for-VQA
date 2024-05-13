import torch


class QuestionDecoder(torch.nn.Module):
    def __init__(
        self,
        n_instructions,
        ninp,
        nhead,
        nhid,
        nlayers,
        dropout=0.1,
    ):
        super(QuestionDecoder, self).__init__()
        # self.text_vocab_embedding = text_vocab_embedding
        self.model_type = "Transformer"
        # self.emb_proj = torch.nn.Linear(text_emb_dim, ninp)
        # self.pos_encoder = PositionalEncoding(ninp, dropout)

        ##################################
        # For Hierarchical Deocding
        ##################################
        # TEXT = GQATorchDataset.TEXT
        self.num_queries = n_instructions
        self.query_embed = torch.nn.Embedding(self.num_queries, ninp)

        decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.coarse_decoder = torch.nn.TransformerDecoder(
            decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp)
        )

        ##################################
        # Decoding
        ##################################
        # decoder_layers = torch.nn.TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        # self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layers, nlayers, norm=torch.nn.LayerNorm(ninp))
        # self.ninp = ninp

        # self.vocab_decoder = torch.nn.Linear(ninp, vocab_size)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, memory):
        ##################################
        # Hierarchical Deocding, first get M instruction vectors
        # in a non-autoregressvie manner
        # Batch_1_Step_1, Batch_1_Step_N, Batch_2_Step_1, Batch_1_Step_N
        # Remember to also update sampling
        ##################################
        true_batch_size = memory.size(1)
        instr_queries = self.query_embed.weight.unsqueeze(1).repeat(
            1, true_batch_size, 1
        )  # [Len, Batch, Dim]
        instr_vectors = self.coarse_decoder(
            tgt=instr_queries, memory=memory, tgt_mask=None
        )  # [ MaxNumSteps, Batch, Dim]
        # instr_vectors_reshape = instr_vectors.permute(1, 0, 2)
        # instr_vectors_reshape = instr_vectors_reshape.reshape( true_batch_size * self.num_queries, -1).unsqueeze(0) # [Len=1, RepeatBatch, Dim]
        # memory_repeat = memory.repeat_interleave(self.num_queries, dim=1) # [Len, RepeatBatch, Dim]
        return instr_vectors

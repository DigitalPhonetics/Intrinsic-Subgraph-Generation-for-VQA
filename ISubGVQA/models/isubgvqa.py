import torch
from .scene_graph_encoder import SceneGraphEncoder
from .question_encoder import QuestionEncoder
from .question_decoder import QuestionDecoder
from .mgat import MGAT
from .att_pooling import GlobalAttention
from datasets.gqa import GQADataset
from torchtext.vocab import GloVe
import copy
from transformers import CLIPModel


class ISubGVQA(torch.nn.Module):
    def __init__(
        self,
        args,
        use_imle=False,
        use_masking=True,
        use_instruction=True,
        use_mgat=False,
        mgat_masks=None,
        use_topk=False,
        interpretable_mode=True,
        concat_instr=False,
        embed_cat=True,
    ):
        super(ISubGVQA, self).__init__()
        self.args = args
        self.n_train_steps = 0
        self.n_valid_steps = 0
        self.use_imle = use_imle
        self.use_instruction = use_instruction
        self.use_masking = use_masking
        self.use_mgat = use_mgat
        self.interpretable_mode = interpretable_mode
        self.concat_instr = concat_instr
        self.embed_cat = embed_cat

        self.general_hidden_dim = args.general_hidden_dim  # 300
        self.scene_graph_encoder = SceneGraphEncoder(
            hidden_dim=self.general_hidden_dim, dist=args.distributed
        )

        self.text_emb_dim = 512
        # self.text_vocab = copy.deepcopy(GQADataset.text_vocab)
        # myvec = GloVe(name="6B", dim=300)
        # vectors = torch.randn((len(self.text_vocab.vocab.itos_), 300))

        # for i, token in enumerate(self.text_vocab.vocab.itos_):
        #     glove_idx = myvec.stoi.get(token)
        #     if glove_idx:
        #         vectors[i] = myvec.vectors[glove_idx]

        # sg_pad_idx = self.text_vocab.vocab.itos_.index("<pad>")
        # self.text_vocab_embedding = torch.nn.Embedding(
        #     len(self.text_vocab.vocab), self.text_emb_dim, padding_idx=sg_pad_idx
        # )
        # self.text_vocab_embedding.weight.data.copy_(vectors)

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.text_vocab_embedding = copy.deepcopy(clip_model.text_model.embeddings)
        del clip_model

        self.question_hidden_dim = self.general_hidden_dim
        n_heads = 6 if self.question_hidden_dim == 300 else 8
        hidden_dim = 512
        self.question_encoder = QuestionEncoder(
            text_vocab_embedding=self.text_vocab_embedding,
            text_emb_dim=self.text_emb_dim,
            ninp=self.text_emb_dim,
            nhead=8,
            nhid=4 * hidden_dim,
            nlayers=4,
            dropout=0.1,
        )
        self.program_decoder = QuestionDecoder(
            n_instructions=args.mgat_layers,
            ninp=self.text_emb_dim,
            nhead=8,
            nhid=4 * hidden_dim,
            nlayers=3,
            dropout=0.1,
        )

        self.gat_seq = MGAT(
            channels=self.general_hidden_dim,
            num_ins=args.mgat_layers,
            use_instr=use_instruction,
            masking_thresholds=mgat_masks,
            use_topk=use_topk,
            interpretable_mode=interpretable_mode,
            concat_instr=concat_instr,
            use_all_instrs=args.use_all_instrs,
            use_global_mask=args.use_global_mask,
            node_classification=args.node_classification,
        )

        ##################################
        # Build Neural Execution Module Pooling Layer
        ##################################
        self.graph_global_attention_pooling = GlobalAttention(
            num_node_features=self.question_hidden_dim,
            num_out_features=self.question_hidden_dim,
        )

        self.qsts_reduction = torch.nn.Sequential(
            torch.nn.Linear(
                self.text_emb_dim * args.mgat_layers,
                self.question_hidden_dim,
            ),
            torch.nn.GELU(),
        )

        self.instr_reduction = torch.nn.Sequential(
            torch.nn.Linear(
                self.text_emb_dim,
                self.question_hidden_dim,
            ),
            torch.nn.GELU(),
        )

        out_classifier_dim = 512
        hid_dim = self.question_hidden_dim * 3

        self.embedding = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, out_classifier_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.2),
        )
        num_short_answer_choices = 1842  # hard coding

        self.logit_fc = torch.nn.Linear(out_classifier_dim, num_short_answer_choices)

        return

    def forward(
        self,
        node_embeddings,
        edge_index,
        edge_embeddings,
        batch,
        questions,
        qsts_att_mask,
        return_masks=False,
        explainer=False,
        explainer_stage=False,
        expl_bypass_x=False,
        scene_graphs=None,
    ):
        ##################################
        # Encode questions
        ##################################
        # [ Len, Batch ] -> [ Len, Batch, self.question_hidden_dim ]
        questions_encoded = self.question_encoder(questions, mask=qsts_att_mask)
        qst_feats = self.program_decoder(memory=questions_encoded)
        mgat_feats_flat = qst_feats.view(
            qst_feats.size(1), int(qst_feats.size(0)), qst_feats.size(2)
        ).flatten(1)
        mgat_language_feat = self.qsts_reduction(mgat_feats_flat)

        if explainer:
            if explainer_stage > 0:
                tmp = node_embeddings.clone()
                node_embeddings = expl_bypass_x
                expl_bypass_x = tmp

        x_encoded, edge_attr_encoded = self.scene_graph_encoder(
            node_embeddings,
            edge_index=edge_index,
            edge_attr=edge_embeddings,
            batch=batch,
            explainer=explainer,
            explainer_stage=explainer_stage,
            gt_scene_graphs=scene_graphs,
        )

        instr_vectors = self.instr_reduction(qst_feats)

        x_mgat, imle_mask, node_logits_layers, hidden_states = self.gat_seq(
            x=x_encoded,
            edge_index=edge_index,
            edge_attr=edge_attr_encoded,
            instr_vectors=instr_vectors[:4],
            global_language_feats=mgat_language_feat,
            batch=batch,
            return_masks=return_masks,
            explainer=explainer,
            explainer_stage=explainer_stage,
            expl_bypass_x=expl_bypass_x,
        )

        mgat_embed, mgat_gate = self.graph_global_attention_pooling(
            x=x_mgat,
            u=mgat_language_feat,
            batch=batch,
            size=None,
            return_mask=return_masks,
            node_mask=imle_mask,
        )
        mgat_feats = torch.cat(
            (mgat_embed, mgat_language_feat, mgat_embed * mgat_language_feat), dim=1
        )
        mgat_feats = self.embedding(mgat_feats)
        mgat_logits = self.logit_fc(mgat_feats)

        if explainer:
            return mgat_logits

        return mgat_logits, imle_mask, mgat_gate, node_logits_layers, None

import copy
import math

import torch
from torchtext.vocab import GloVe
from transformers import CLIPModel

from ..datasets.gqa import GQADataset
from .att_pooling import GlobalAttention
from .mgat import MGAT
from .question_decoder import QuestionDecoder
from .question_encoder import QuestionEncoder
from .scene_graph_encoder import SceneGraphEncoder
from ..sampling.methods.simple_scheme import EdgeSIMPLEBatched


class ISubGVQA(torch.nn.Module):
    """
    ISubGVQA is a PyTorch neural network module designed for Visual Question Answering (VQA) tasks.
    It integrates various components such as scene graph encoding, question encoding, and multi-head
    graph attention networks to process and interpret visual and textual data.
    Args:
        args (Namespace): Configuration arguments.
        use_imle (bool, optional): Flag to use IMLE. Defaults to False.
        use_masking (bool, optional): Flag to use masking. Defaults to True.
        use_instruction (bool, optional): Flag to use instruction. Defaults to True.
        use_mgat (bool, optional): Flag to use multi-head graph attention. Defaults to False.
        mgat_masks (optional): Masks for MGAT. Defaults to None.
        use_topk (bool, optional): Flag to use top-k sampling. Defaults to False.
        interpretable_mode (bool, optional): Flag for interpretable mode. Defaults to True.
        concat_instr (bool, optional): Flag to concatenate instructions. Defaults to False.
        embed_cat (bool, optional): Flag to embed categories. Defaults to True.
    Attributes:
        args (Namespace): Configuration arguments.
        n_train_steps (int): Number of training steps.
        n_valid_steps (int): Number of validation steps.
        use_imle (bool): Flag to use IMLE.
        use_instruction (bool): Flag to use instruction.
        use_masking (bool): Flag to use masking.
        use_mgat (bool): Flag to use multi-head graph attention.
        interpretable_mode (bool): Flag for interpretable mode.
        concat_instr (bool): Flag to concatenate instructions.
        embed_cat (bool): Flag to embed categories.
        text_sampling (bool): Flag for text sampling.
        general_hidden_dim (int): General hidden dimension size.
        scene_graph_encoder (SceneGraphEncoder): Scene graph encoder module.
        text_emb_dim (int): Text embedding dimension size.
        text_vocab_embedding (torch.nn.Embedding): Text vocabulary embedding.
        question_hidden_dim (int): Question hidden dimension size.
        question_encoder (QuestionEncoder): Question encoder module.
        text_sampler (EdgeSIMPLEBatched, optional): Text sampler module.
        qsts_att_keys (torch.nn.Sequential, optional): Attention keys for questions.
        qsts_att_query (torch.nn.Sequential, optional): Attention query for questions.
        program_decoder (QuestionDecoder): Program decoder module.
        gat_seq (MGAT): Multi-head graph attention module.
        graph_global_attention_pooling (GlobalAttention): Global attention pooling layer.
        qsts_reduction (torch.nn.Sequential): Question reduction layer.
        instr_reduction (torch.nn.Sequential): Instruction reduction layer.
        embedding (torch.nn.Sequential): Embedding layer.
        logit_fc (torch.nn.Linear): Final classification layer.
    Methods:
        forward(node_embeddings, edge_index, edge_embeddings, batch, questions, qsts_att_mask,
                return_masks=False, explainer=False, explainer_stage=False, expl_bypass_x=False,
                scene_graphs=None):
            Forward pass of the model.
            Args:
                node_embeddings (torch.Tensor): Node embeddings.
                edge_index (torch.Tensor): Edge indices.
                edge_embeddings (torch.Tensor): Edge embeddings.
                batch (torch.Tensor): Batch indices.
                questions (torch.Tensor): Encoded questions.
                qsts_att_mask (torch.Tensor): Attention mask for questions.
                return_masks (bool, optional): Flag to return masks. Defaults to False.
                explainer (bool, optional): Flag for explainer mode. Defaults to False.
                explainer_stage (bool, optional): Stage for explainer. Defaults to False.
                expl_bypass_x (bool, optional): Bypass for explainer. Defaults to False.
                scene_graphs (optional): Scene graphs. Defaults to None.
            Returns:
                torch.Tensor: Model logits.
                torch.Tensor: IMLE mask.
                torch.Tensor: MGAT gate.
                torch.Tensor: Node logits layers.
                torch.Tensor: Mask text.
    """

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
        self.text_sampling = args.text_sampling

        self.general_hidden_dim = args.general_hidden_dim  # 300
        self.scene_graph_encoder = SceneGraphEncoder(
            hidden_dim=self.general_hidden_dim, dist=args.distributed
        )

        self.text_emb_dim = 512

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
        if args.text_sampling:
            self.text_sampler = EdgeSIMPLEBatched(
                k=args.mgat_layers,
                device="cuda",
                policy="edge_candid",
            )
            self.qsts_att_keys = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
            )
            self.qsts_att_query = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.GELU(),
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
            sampler_type=args.sampler_type,
            sample_k=args.sample_k,
            nb_samples=args.nb_samples,
            alpha=args.alpha,
            beta=args.beta,
            tau=args.tau,
        )

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
        mask_text = None
        questions_encoded = self.question_encoder(questions, mask=qsts_att_mask)
        if self.text_sampling:
            qsts_keys = self.qsts_att_keys(questions_encoded)
            qsts_queries = self.qsts_att_query(questions_encoded)
            qsts_logits = torch.bmm(
                qsts_keys.permute(1, 0, 2), qsts_queries.permute(1, 2, 0)
            ).sum(-1) / math.sqrt(questions_encoded.size(-1))
            self.text_sampler.device = qsts_logits.device
            mask_text, _ = self.text_sampler(
                qsts_logits.unsqueeze(-1), train=self.training
            )
            questions_encoded = (
                questions_encoded.permute(1, 0, 2) * mask_text.squeeze(0)
            ).permute(1, 0, 2)

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

        return mgat_logits, imle_mask, mgat_gate, node_logits_layers, mask_text

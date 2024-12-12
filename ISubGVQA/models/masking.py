import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_add

from ..sampling.methods.aimle import aimle

from ..sampling.methods.gumbel_scheme import GumbelSampler
from ..sampling.methods.imle_scheme import IMLEScheme
from ..sampling.methods.noise import GumbelDistribution
from ..sampling.methods.simple import Layer
from ..sampling.methods.simple_scheme import EdgeSIMPLEBatched
from ..sampling.methods.target import TargetDistribution
from ..sampling.methods.target_aimle import (
    TargetDistribution as TargetDistributionAIMLE,
)
from ..sampling.methods.target_aimle import AdaptiveTargetDistribution
from ..sampling.methods.wrapper import imle
from ..utils.topk import topk


class MaskingModel(torch.nn.Module):
    def __init__(
        self,
        dim_nodes,
        dim_questions,
        masking_threshold=0.3,
        use_topk=False,
        sample_k=None,
        sampler_type=None,
        nb_samples=1,
        alpha=1.0,
        beta=10.0,
        tau=1.0,
    ):
        super(MaskingModel, self).__init__()
        # self.gate_nn = Seq(Linear(channels, channels), GELU(), Linear(channels, 1))
        # self.node_nn = Seq(Linear(num_node_features, channels), GELU(), Linear(channels, channels))
        # self.ques_nn = Seq(Linear(channels, channels), GELU(), Linear(channels, channels))
        self.use_topk = use_topk
        self.sample_k = sample_k
        self.sampler_type = sampler_type
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

        print(f"using {sampler_type} as sample method")
        match sampler_type:
            case "imle":
                self.sampler_train, self.sampler_val = get_imle_samplers(
                    sample_k=sample_k,
                    device="cuda",
                    nb_samples=nb_samples,
                    alpha=alpha,
                    beta=beta,
                    tau=tau,
                )
            case "aimle":
                self.sampler_train, self.sampler_val = get_aimle_samplers(
                    sample_k=sample_k,
                    device="cuda",
                    nb_samples=nb_samples,
                    alpha=alpha,
                    tau=tau,
                )
            case "simple":
                self.sampler = EdgeSIMPLEBatched(
                    k=sample_k,  # (self.masking_threshold * num_nodes.to(torch.float)).ceil().to(torch.long),
                    device="cuda",
                    # val_ensemble=imle_configs.num_val_ensemble,
                    # train_ensemble=imle_configs.num_train_ensemble,
                    policy="edge_candid",
                    # logits_activation=imle_configs.logits_activation,
                )
            case "gumbel":
                self.sampler = GumbelSampler(
                    k=sample_k, policy="edge_candid", train_ensemble=1, val_ensemble=1
                )

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

            # imle_solver = imle(
            #     topk_sampling,
            #     target_distribution=TargetDistribution(alpha=0.0, beta=10.0),
            #     noise_distribution=SumOfGammaNoiseDistribution(k=3, nb_iterations=100),
            #     nb_samples=1,
            #     input_noise_temperature=1.0,
            #     target_noise_temperature=1.0,
            # )
            # gate = imle_solver(
            #     gate,
            #     self.masking_threshold,
            #     batch,
            #     edge_index,
            #     torch.tensor(gate.shape[0]),
            # )

            # num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
            gate, mask = to_dense_batch(gate, batch)
            if self.sampler_type in ["imle", "aimle"]:
                output = (
                    self.sampler_train(gate)
                    if self.training
                    else self.sampler_val(gate)
                )
                gate = (
                    output[0].squeeze(0)[mask]
                    if self.sampler_type == "imle"
                    else output[mask]
                )
            else:
                output, _ = self.sampler(gate, train=self.training)
                gate = output.squeeze(0)[mask]

            # simple_layer = Layer(n=6, k=3, device=gate.device)
            # simple_layer(to_dense_batch(gate.squeeze(), batch)[0], k=1)

            # gate = topk_sampling(gate, self.masking_threshold, batch)
            # gate = gate.squeeze(-1)
            # gate_indices = topk(gate.squeeze(), self.masking_threshold, batch=batch)
            # neg_indices = torch.tensor(
            #     [
            #         i
            #         for i, e in enumerate(range(batch.size(0)))
            #         if i not in gate_indices
            #     ],
            #     device=x.device,
            # )
            # gate.index_fill_(0, gate_indices, 1)
            # if neg_indices.nelement() != 0:
            #     gate.index_fill_(0, neg_indices, 0)
        else:
            gate = F.dropout(gate, p=0.2, training=self.training)
            gate = F.sigmoid(gate)
            gate = (gate > 0.5).to(dtype=gate.dtype)
        return gate


def topk_sampling(gate, masking_threshold, batch):
    gate_indices = topk(gate.squeeze(), masking_threshold, batch=batch)
    neg_indices = torch.tensor(
        [i for i, e in enumerate(range(batch.size(0))) if i not in gate_indices],
        device=gate.device,
    )
    gate.index_fill_(0, gate_indices, 1)
    if neg_indices.nelement() != 0:
        gate.index_fill_(0, neg_indices, 0)
    return gate


def get_imle_samplers(
    sample_k, beta=10, alpha=1.0, tau=1.0, noise_scale=0.3, nb_samples=1, device=None
):
    imle_scheduler = IMLEScheme(
        "edge_candid",
        sample_k,
        1,
        1,
    )

    @imle(
        target_distribution=TargetDistribution(alpha=alpha, beta=beta),
        noise_distribution=GumbelDistribution(0.0, noise_scale, device),
        nb_samples=nb_samples,
        input_noise_temperature=tau,
        target_noise_temperature=tau,
    )
    def imle_train_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)

    @imle(
        target_distribution=None,
        noise_distribution=GumbelDistribution(0.0, noise_scale, device),
        nb_samples=nb_samples,
        input_noise_temperature=tau if nb_samples > 1 else 0.0,
        # important
        target_noise_temperature=tau,
    )
    def imle_val_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)

    return imle_train_scheme, imle_val_scheme


def get_aimle_samplers(
    sample_k, alpha=1.0, tau=1.0, noise_scale=0.3, nb_samples=1, device=None
):
    imle_scheduler = IMLEScheme(
        "edge_candid",
        sample_k,
        1,
        1,
    )

    @aimle(
        target_distribution=AdaptiveTargetDistribution(
            initial_alpha=alpha, initial_beta=0.0
        ),
        noise_distribution=GumbelDistribution(0.0, noise_scale, device),
        nb_samples=nb_samples,
        theta_noise_temperature=tau,
        target_noise_temperature=tau,
        symmetric_perturbation=True,
    )
    def imle_train_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)

    @aimle(
        target_distribution=None,
        noise_distribution=GumbelDistribution(0.0, noise_scale, device),
        nb_samples=nb_samples,
        theta_noise_temperature=1.0 if nb_samples > 1 else tau,
        # important
        target_noise_temperature=tau,
        symmetric_perturbation=True,
    )
    def imle_val_scheme(logits: torch.Tensor):
        return imle_scheduler.torch_sample_scheme(logits)

    return imle_train_scheme, imle_val_scheme

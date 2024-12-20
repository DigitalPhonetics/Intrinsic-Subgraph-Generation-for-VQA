import torch
import numpy as np
from .deterministic_scheme import (
    rewire_global_undirected,
    rewire_global_directed,
    select_from_edge_candidates,
)

EPSILON = np.finfo(np.float32).tiny
LARGE_NUMBER = 1.0e10


class GumbelSampler(torch.nn.Module):
    def __init__(
        self, k, train_ensemble, val_ensemble, tau=0.1, hard=True, policy=None
    ):
        super(GumbelSampler, self).__init__()
        self.policy = policy
        self.k = k
        self.hard = hard
        self.tau = tau
        self.adj = None  # for potential usage
        self.train_ensemble = train_ensemble
        self.val_ensemble = val_ensemble

    def forward(self, scores, train=True):
        repeat_sample = self.train_ensemble if train else self.val_ensemble
        if self.policy == "global_directed":
            bsz, Nmax, _, ensemble = scores.shape
            local_k = min(
                self.k,
                Nmax**2 - torch.unique(self.adj[0], return_counts=True)[1].max().item(),
            )
            scores[self.adj] = (
                scores[self.adj] - LARGE_NUMBER
            )  # avoid selecting existing edges & self loops
            flat_scores = scores.permute((0, 3, 1, 2)).reshape(bsz * ensemble, Nmax**2)
        elif self.policy == "global_undirected":
            bsz, Nmax, _, ensemble = scores.shape
            local_k = min(
                self.k,
                (Nmax * (Nmax - 1)) // 2
                - torch.unique(self.adj[0], return_counts=True)[1].max().item(),
            )
            scores[self.adj] = (
                scores[self.adj] - LARGE_NUMBER
            )  # avoid selecting existing edges & self loops
            scores = scores + scores.transpose(1, 2)  # make symmetric
            triu_idx = np.triu_indices(Nmax, k=1)
            flat_scores = (
                scores[:, triu_idx[0], triu_idx[1], :]
                .permute((0, 2, 1))
                .reshape(bsz * ensemble, -1)
            )
        elif self.policy == "edge_candid":
            bsz, Nmax, ensemble = scores.shape
            flat_scores = scores.permute((0, 2, 1)).reshape(bsz * ensemble, Nmax)
            local_k = min(self.k, Nmax)
        else:
            raise NotImplementedError

        # sample several times with
        flat_scores = flat_scores.repeat(repeat_sample, 1)

        m = torch.distributions.gumbel.Gumbel(
            flat_scores.new_zeros(flat_scores.shape),
            flat_scores.new_ones(flat_scores.shape),
        )
        g = m.sample()
        flat_scores = flat_scores + g

        # continuous top k
        khot = flat_scores.new_zeros(flat_scores.shape)
        onehot_approx = flat_scores.new_zeros(flat_scores.shape)
        for i in range(local_k):
            khot_mask = torch.max(
                1.0 - onehot_approx, torch.tensor([EPSILON], device=flat_scores.device)
            )
            flat_scores = flat_scores + torch.log(khot_mask)
            onehot_approx = torch.nn.functional.softmax(flat_scores / self.tau, dim=1)
            khot = khot + onehot_approx

        if self.hard:
            # straight through
            khot_hard = khot.new_zeros(khot.shape)
            val, ind = torch.topk(khot, local_k, dim=1)
            khot_hard = khot_hard.scatter_(1, ind, 1)
            res = khot_hard - khot.detach() + khot
        else:
            res = khot

        if self.policy == "global_directed":
            new_mask = res.reshape(repeat_sample, bsz, ensemble, Nmax, Nmax).permute(
                (0, 1, 3, 4, 2)
            )
        elif self.policy == "global_undirected":
            res = res.reshape(repeat_sample, bsz, ensemble, -1).permute((0, 1, 3, 2))
            new_mask = scores.new_zeros((repeat_sample,) + scores.shape)
            new_mask[:, :, triu_idx[0], triu_idx[1], :] = res
            new_mask = new_mask + new_mask.transpose(2, 3)
        elif self.policy == "edge_candid":
            new_mask = res.reshape(repeat_sample, bsz, ensemble, Nmax).permute(
                (0, 1, 3, 2)
            )
        else:
            raise NotImplementedError
        return new_mask, None

    @torch.no_grad()
    def validation(self, scores):
        if self.val_ensemble == 1:
            if self.policy == "global_directed":
                mask = rewire_global_directed(scores, self.k, self.adj)
            elif self.policy == "global_undirected":
                mask = rewire_global_undirected(scores, self.k, self.adj)
            elif self.policy == "edge_candid":
                mask = select_from_edge_candidates(scores, self.k)
            else:
                raise NotImplementedError

            return mask[None], None
        else:
            return self.forward(scores, False)

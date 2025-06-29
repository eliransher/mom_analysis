import os.path
import torch
import pickle as pkl
import numpy as np
import pandas as pd
import time
import random


class MomentMatcherBase(object):
    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lr_gamma=.9, weights = None):
        self.weights = weights
        self.k = ph_size
        self.n = n_replica
        self.lr = lr
        self.lr_gamma = lr_gamma
        self.n_epochs = num_epochs
        self.params = None
        self.fit_info = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Starting with device: {self.device}")

    def fit(self, target_ms, min_loss, stop=None):
        # init
        best_loss = []
        self._init()
        optimizer = torch.optim.Adam(self.params, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)

        # train loop
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            loss, extended_loss_info = self._loss(target_ms.to(self.device))
            loss.backward()
            optimizer.step()


            losses = extended_loss_info["per_replica"]
            best_replica_loss = torch.min(losses[~torch.isnan(losses)])
            best_replica_ix = torch.argmin(losses[~torch.isnan(losses)])
            still_alive_count = torch.sum(torch.isfinite(losses))
            best_loss.append(best_replica_loss)

            if epoch % 1000 == 0 and epoch > 0:
                scheduler.step()

            if (epoch % 1000 == 0 ) & (epoch > 2999):
                if best_replica_loss > 1:
                    print('########## breaking - after 3000 not good #########')
                    break
            if (epoch % 1000 == 0 ) & (epoch > 4999):
                if 100*torch.abs((best_loss[-5000]-best_loss[-1])/best_loss[-1]) < 0.1:
                    self.fit_info = extended_loss_info
                    print('########## breaking - stuck in local minumum #########')
                    break

            if  (epoch % 1000 == 0 ) &(best_replica_loss > 0.001) & (epoch > 4999):
                self.fit_info = extended_loss_info
                print('########## breaking - after 5000 not good #########')
                break

            if  (epoch % 1000 == 0 ) &(best_replica_loss > 0.0001) & (epoch > 14999):
                self.fit_info = extended_loss_info
                print('########## breaking - after 15000 not good #########')
                break



            if (epoch % 1000 == 0 and epoch > 0)  or best_replica_loss < min_loss or epoch == self.n_epochs-1:
                print(f"===== Epoch: {epoch} =====")
                print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']}")
                print(f"Overall loss: {loss}")
                print(f"Still alive: {still_alive_count}")
                print(f"Best replica (ix={best_replica_ix}): {best_replica_loss}")
                self.fit_info = extended_loss_info

            if best_replica_loss < min_loss:
                self.fit_info = extended_loss_info
                break

            if stop is not None:
                for level in stop:
                    if epoch == level["epoch"]:
                        # structure: {"when":epoch, "keep_fraction": num}
                        n_keep = int(level["keep_fraction"] * self.n)
                        ix_keep = torch.argsort(losses)[:n_keep]
                        self.params = tuple(el[ix_keep].clone().detach().to(self.device).requires_grad_(True) for el in self.params)
                        self.n = self.params[0].shape[0]
                        optimizer = torch.optim.Adam(self.params, lr=self.lr)
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)


    @staticmethod
    def _compute_moments(a, T, n_moments, device):
        n, k, _ = T.shape
        m = []
        T_in = torch.inverse(T)
        T_powers = torch.eye(k).expand(n, k, k).to(device)
        signed_factorial = 1.
        one = torch.ones(k).to(device)

        for i in range(1, n_moments+1):
            signed_factorial *= -i
            T_powers = torch.matmul(T_powers, T_in)  # now T_powers is T^(-i)
            current_moment = signed_factorial * torch.einsum('bi,bij,j->b', a, T_powers, one)
            m.append(current_moment)

        return torch.stack(m).T

    def _loss(self, target_ms):
        a, T = self._make_phs_from_params()
        a = a.to(self.device)
        T = T.to(self.device)
        ms = self._compute_moments(a, T, n_moments=len(target_ms), device=self.device)
        weighted_error = self.weights*(ms - target_ms.to(self.device)) / target_ms.to(self.device)
        per_replica_loss = torch.mean(weighted_error ** 2, dim=1)

        # Save all loss values
        extended_loss_info = {"per_replica": per_replica_loss.detach()}

        # Compute total loss without nan/inf (dead) replicas
        mask = ~torch.isnan(per_replica_loss) & ~torch.isinf(per_replica_loss)
        loss = per_replica_loss[mask].to(torch.float64).mean()

        return loss, extended_loss_info

    def get_best_after_fit(self):
        all_loss = self.fit_info["per_replica"]
        best_instance = torch.nan_to_num(all_loss, nan=float('inf')).argmin()
        a_all, T_all = self._make_phs_from_params()
        return a_all[best_instance], T_all[best_instance]


class GeneralPHMatcher(MomentMatcherBase):

    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lr_gamma=.9,
                 normalize_m1=True,
                 init_drop=None, weights=None):

        super().__init__(ph_size, n_replica, lr, num_epochs, lr_gamma)
        self.normalize_m1 = normalize_m1
        self.init_drop = init_drop
        self.weights = weights

    def _init(self):
        device = self.device

        # sample P uniform on hte n-1 simplex (after the softmax transformation)
        ones = torch.ones(self.k)
        qs = torch.distributions.Dirichlet(ones).sample((self.n, self.k))
        ps = torch.log(qs).to(self.device)

        lambdas = torch.empty(self.n, self.k, requires_grad=False).to(self.device)
        lambdas.data = torch.rand(self.n, self.k).to(device) ** (.5) # self.ls

        alpha = torch.distributions.Dirichlet(ones).sample((self.n, 1)).squeeze(1).to(self.device)

        # calculate 1st moment and normalize lambdas
        if self.normalize_m1:
            self.params = alpha, lambdas, ps
            a, T = self. _make_phs_from_params()
            m1 = self._compute_moments(a, T, n_moments=1, device=device)
            lambdas.data = lambdas.data * m1**0.5

        # zero out some P values
        if self.init_drop is not None:
            if self.init_drop == "uniform":

                # first sample for each replica the p of drop, then sample the drop mask
                drop_prob = torch.rand(self.n)
                drop_mask = torch.empty(self.n, self.k, self.k)

                for i in range(self.n):
                    sub_mask = torch.bernoulli(torch.full((k,k), drop_prob[i]))
                    drop_mask[i] = sub_mask
            else:
                # all use the same proportion of drop
                drop_mask = torch.bernoulli(torch.full(ps.shape, self.init_drop))
                ps[drop_mask == 1] = 0.

        ps.requires_grad_(True)
        lambdas.requires_grad_(True)
        alpha.requires_grad_(True)

        self.params = alpha, lambdas, ps

    def _make_phs_from_params(self):
        alpha, lambdas, ps = self.params
        a = torch.nn.functional.softmax(alpha, dim=1)
        ls = lambdas ** 2
        lambda_rows = ls.unsqueeze(-1).expand(-1, -1, self.k)
        p = torch.nn.functional.softmax(ps, 2)

        diagonals = -torch.diagonal(p, dim1=1, dim2=2) - 1
        diagonals_back_to_3D = torch.diag_embed(diagonals)

        T = (p + diagonals_back_to_3D) * lambda_rows
        return a, T


class HyperErlangMatcher(MomentMatcherBase):
    def __init__(self, block_sizes, n_replica=10, lr=1e-2, num_epochs=1000, lr_gamma=.9, weights = None, sort_init=True,
                 normalize_m1=True):
        super().__init__(sum(block_sizes), n_replica, lr, num_epochs, lr_gamma)

        self.weights = weights
        self.block_sizes = block_sizes
        self.n_blocks = len(block_sizes)
        self.sort_init = sort_init
        self.normalize_m1 = normalize_m1
     

    def _init(self):
        device = self.device

        lam = torch.rand(self.n, self.n_blocks).to(self.device)

        if self.sort_init:
            lam, _ = lam.sort(dim=1)

        lambdas = torch.empty(self.n, self.n_blocks, requires_grad=False).to(self.device)
        lambdas.data = lam

        ps = torch.randn(self.n, self.n_blocks, requires_grad=False).to(self.device)

        ones = torch.ones(self.n_blocks).to(self.device)
        alpha = torch.distributions.Dirichlet(ones).sample((self.n, 1)).squeeze(1).to(self.device)

        if self.normalize_m1:
            self.params = alpha, lambdas, ps
            a, T = self._make_phs_from_params()
            a = a.to(self.device)
            T = T.to(self.device)
            m1 = self._compute_moments(a, T, n_moments=1, device=device)
            lambdas.data = lambdas.data * m1 ** 0.5

        lambdas.requires_grad_(True)
        ps.requires_grad_(True)
        alpha.requires_grad_(True)
        self.params = alpha, lambdas, ps

    def _make_phs_from_params(self):
        alpha, lambdas, ps = self.params

        # Make a by softmax and adding in zeros
        a = torch.nn.functional.softmax(alpha, dim=1)
        aa = torch.zeros(self.n, self.k)
        cuma = torch.cumsum(torch.tensor(self.block_sizes), dim=0)

        for i in range(self.n_blocks):
            aa[:, cuma[i] - self.block_sizes[i]] = a[:, i]

        # lambda and ps are repeated according to block sizes
        ls = torch.cat([lambdas[:, [i]].repeat(1, bs) for i, bs in enumerate(self.block_sizes)], dim=1) ** 2
        ps = torch.cat([ps[:, [i]].repeat(1, bs) for i, bs in enumerate(self.block_sizes)], dim=1)
        ps = torch.sigmoid(ps)

        T = torch.diag_embed(-ls) + torch.diag_embed(ps[:, :-1] * ls[:, :-1], offset=1)

        return aa, T

class CoxianPHMatcher(MomentMatcherBase):
    def __init__(self, ph_size, n_replica=10, lr=1e-4, num_epochs=1000, lr_gamma=.9, normalize_m1=True, sort_init=True, weights=None):
        super().__init__(ph_size, n_replica, lr, num_epochs, lr_gamma, weights)
        self.sort_init = sort_init
        self.normalize_m1 = normalize_m1
        self.weights = weights

    def _init(self):
        device = self.device

        lam = torch.rand(self.n, self.k).to(self.device)

        if self.sort_init:
            lam, _ = lam.sort(dim=1)

        lambdas = torch.empty(self.n, self.k, requires_grad=False).to(self.device)
        lambdas.data = lam

        ps = torch.randn(self.n, self.k-1, requires_grad=False).to(self.device)

        if self.normalize_m1:
            self.params = lambdas, ps
            a, T = self._make_phs_from_params()
            m1 = self._compute_moments(a, T, n_moments=1, device=device)
            lambdas.data = lambdas.data * m1 ** 0.5

        lambdas.requires_grad_(True)
        ps.requires_grad_(True)
        self.params = lambdas, ps

    def _make_phs_from_params(self):
        lambdas, ps = self.params
        ls = lambdas ** 2

        p = torch.sigmoid(ps)

        a = torch.zeros(self.n, self.k).to(self.device)
        a[:, 0] = 1.

        T = torch.diag_embed(-ls) + torch.diag_embed(p * ls[:, :-1], offset=1)

        return a, T
import torch
import torch.nn as nn
import einops

from network_utils import inverse_softplus, fill_triangular, diag_bijector

def get_gmm_head(n_dim, n_components, init_std, minimal_std, type='full', device='cuda'):
    if type == 'full':
        return FullGMMHead(n_dim, n_components, init_std, minimal_std, device=device)
    else:
        raise NotImplementedError(f"Unknown GMM head type {type}")


class AbstractGaussianHead(nn.Module):
    def __init__(self, n_dim, init_std, minimal_std, device='cuda'):
        super(AbstractGaussianHead, self).__init__()
        self.device = device
        self.n_dim = n_dim
        self.input_dim = 2 * n_dim
        self.minimal_std = torch.tensor(minimal_std, device=device)
        self.init_std = torch.tensor(init_std, device=device)
        self.diag_activation = nn.Softplus()
        self.diag_activation_inv = inverse_softplus
        ##TODO: check if this is correct
        self._pre_activation_shift = inverse_softplus(init_std - minimal_std)

    def forward(self, mean, chol, train=True):
        raise NotImplementedError

    @staticmethod
    def log_prob(x, mean, chol):
        return torch.distributions.MultivariateNormal(mean, scale_tril=chol, validate_args=False).log_prob(x)

    @staticmethod
    def rsample(mean, chol, n=1):
        return torch.distributions.MultivariateNormal(mean, scale_tril=chol, validate_args=False).rsample((n,))

    @staticmethod
    def entropy(mean, chol):
        return torch.distributions.MultivariateNormal(mean, scale_tril=chol, validate_args=False).entropy()

    def get_device(self, device: torch.device):
        self.device = device

    def get_params(self):
        return self.parameters()


class AbstractGMMHead(AbstractGaussianHead):
    def __init__(self, n_dim, n_components, init_std, minimal_std, device='cuda'):
        super(AbstractGMMHead, self).__init__(n_dim, init_std, minimal_std, device=device)
        self.n_components = n_components
        self.flat_mean_dim = n_dim * n_components
        self.flat_chol_dim = 2 * n_dim * n_components

    def forward(self, flat_mean, flat_chol, train=True):
        raise NotImplementedError

    @staticmethod
    def gmm_log_prob(x, means, chols, gating):
        comps = torch.distributions.MultivariateNormal(means, scale_tril=chols)
        gmm = torch.distributions.MixtureSameFamily(gating, comps)
        return gmm.log_prob(x)

    @staticmethod
    def gmm_sample(means, chols, gating, n=1):
        comps = torch.distributions.MultivariateNormal(means, scale_tril=chols)
        gmm = torch.distributions.MixtureSameFamily(gating, comps)
        return gmm.sample((n,))

    @staticmethod
    def log_responsibilities(pred_means, pred_chols, pred_gatings, samples):
        """
        b -- state batch
        c -- the number of components
        v -- the number of vi samples
        a -- action dimension
        """
        c = pred_means.shape[1]
        v = samples.shape[-2]

        ### pred_means: (b, c, a)
        ### pred_chols: (b, c, a, a)
        pred_means = pred_means[:, None, :, None, ...].repeat(1, 1, 1, v, 1)
        pred_chols = pred_chols[:, None, :, None, ...].repeat(1, 1, 1, v, 1, 1)

        samples = samples.unsqueeze(2).repeat(1, 1, c, 1, 1)

        ### samples: (b, c, c, v, a)
        ### log_probs_cmps: (b, c, c, v)
        log_probs_cmps = AbstractGaussianHead.log_prob(samples, pred_means, pred_chols)

        ### log_probs: (b, c, v)
        log_probs = log_probs_cmps.clone()
        log_probs = torch.einsum('ijj...->ij...', log_probs)

        log_gating = torch.log(pred_gatings)

        probs_cmps = log_probs_cmps.exp()

        margin = torch.einsum('ijkl,ik->ijl', probs_cmps, pred_gatings)

        log_margin = torch.log(margin + 1e-8)

        return log_probs + log_gating.unsqueeze(-1) - log_margin


class FullGMMHead(AbstractGMMHead):
    def __init__(self, n_dim, n_components, init_std=1., minimal_std=1e-3, device='cuda'):
        super(FullGMMHead, self).__init__(n_dim, n_components, init_std, minimal_std, device=device)
        self.flat_mean_dim = n_dim * n_components
        self.flat_chol_dim = n_dim * (n_dim + 1) // 2 * n_components

    def forward(self, flat_means, flat_chols, train=True):
        assert flat_means.shape[-1] == self.flat_mean_dim, f"Expected {self.flat_mean_dim} but got {flat_means.shape[-1]}"
        assert flat_chols.shape[-1] == self.flat_chol_dim, f"Expected {self.flat_chol_dim} but got {flat_chols.shape[-1]}"
        means = einops.rearrange(flat_means, '... (n d) -> ... n d', n=self.n_components, d=self.n_dim)
        chols = einops.rearrange(flat_chols, '... (n d) -> ... n d', n=self.n_components)
        chols = fill_triangular(chols)
        chols = diag_bijector(lambda z: self.diag_activation(z + self._pre_activation_shift) + self.minimal_std, chols)
        return means, chols


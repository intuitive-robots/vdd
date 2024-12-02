from vdd.score_functions.score_base import ScoreFunction

import torch as ch
import torch.distributions as D
import torch.nn.functional as F

import matplotlib.pyplot as plt

from vdd.score_functions.score_utils import plot_2d_gaussians, plot_2d_gaussians_color_map

class GMMScoreFunction(ScoreFunction):
    def __init__(self, model=None, means=ch.Tensor, chols=ch.Tensor, prior=None, device='cuda'):
        super().__init__(model)
        self.dim = means.shape[-1]
        self.n_components = means.shape[0]
        self.means = means.to(device)
        self.chols = chols.to(device)
        assert self.means.shape == self.chols.shape[:-1], f"Means shape: {self.means.shape}, Chols shape: {self.chols.shape}"
        self.prior = prior if prior is not None else ch.ones(means.shape[0]) / self.n_components
        self.prior = self.prior.to(device)
        self.device = device

    def log_probability(self, x):
        gating = self.prior.view(1, -1).repeat(x.shape[0], 1)
        gating_dist = D.Categorical(gating)
        cmps = D.MultivariateNormal(self.means, scale_tril=self.chols, validate_args=False)
        gmm = D.MixtureSameFamily(gating_dist, cmps)
        return gmm.log_prob(x)

    def sample(self, n: int = 1):
        gating = D.Categorical(self.prior)
        comps = D.MultivariateNormal(self.means, scale_tril=self.chols, validate_args=False)
        gmm = D.MixtureSameFamily(gating, comps)
        return gmm.sample((n,))

    def __call__(self, samples, states=None, score_goals=False, iter=None, is_vision=False):
        with ch.enable_grad():
            samples = samples.clone().detach().requires_grad_(True)
            samples.retain_grad()
            log_probs = self.log_probability(samples)
            log_probs = log_probs.sum()/samples.shape[0]
            log_probs.backward()
        return samples.grad, ch.zeros(samples.shape[0])

    def visualize_cmps(self, ax=None):
        cmp_means = self.means.clone().cpu()
        cmp_chols = self.chols.clone().cpu()
        if ax is None:
            fig, ax = plt.subplots(1, 1)
            plot_2d_gaussians(cmp_means, cmp_chols, ax, title="GT GMM")
            ax.set_aspect('equal')
            plt.show()
        else:
            plot_2d_gaussians(cmp_means, cmp_chols, ax, title="GT GMM")
            ax.set_aspect('equal')

    def visualize_gradient_field(self, n=20, x_range=[-1, 1], y_range=[-1, 1], ax=None):
        raw_x, raw_y = ch.meshgrid(ch.linspace(x_range[0], x_range[1], n), ch.linspace(x_range[0], y_range[1], n))
        raw_x = raw_x.to(self.device)
        raw_y = raw_y.to(self.device)
        grid_actions = ch.stack([raw_x, raw_y], dim=-1).view(-1, 2)
        scores = self(grid_actions, None)[0]
        scores = scores.view(n, n, 2).cpu()
        u = scores[..., 0]
        v = scores[..., 1]

        # fig, ax = plt.subplots(1, 1)
        ax.quiver(raw_x.cpu(), raw_y.cpu(), u, v, color="white"
                                                        "")
        # self.visualize_cmps(ax=ax)
        # plt.show()

    def visualize_grad_and_cmps(self, x_range=[-1, 1], y_range=[-1, 1], n=20):
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)

        # self.visualize_cmps(ax=ax)
        plot_2d_gaussians_color_map(self.means.cpu(), self.chols.cpu(), ax,
                                    x_range=x_range, y_range=y_range,
                                    title="GT GMM")
        self.visualize_gradient_field(n=n, x_range=x_range, y_range=y_range,
                                      ax=ax)
        return fig, ax

    @staticmethod
    def generate_random_params(n_components, dim):
        means = ch.rand(n_components, dim) * 2.0 - 1.0
        chols = ch.rand(n_components, dim, dim)
        chols = ch.tril(chols, diagonal=-1)
        diag = ch.rand(n_components, dim)
        diag = F.softplus(diag) + 1e-4
        chols = chols + ch.diag_embed(diag)
        chols = 0.15 * chols
        return means, chols


if __name__ == "__main__":
    device = 'cuda'
    means, chols = GMMScoreFunction.generate_random_params(5, 2)
    print(chols.shape)

    score_function = GMMScoreFunction(means=means, chols=chols, device=device)
    samples = ch.rand(100, 2).to(device)
    # score_function.visualize_cmps()
    with ch.no_grad():
        scores = score_function(samples, None)

    fig, ax = plt.subplots(1, 1)
    score_function.visualize_cmps(ax=ax)
    score_function.visualize_gradient_field(ax=ax)
    plt.show()
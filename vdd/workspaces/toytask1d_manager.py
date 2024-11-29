import torch as ch
import torch.utils.data as Data
import numpy as np

from vi.experiment_managers.base_manager import BaseManager
from vi.score_functions.gating_toytask_score import ToyTaskScoreFunction

import matplotlib.pyplot as plt

def plot_1d_gaussians(means, stds, ax, title=""):
    x = ch.linspace(-2.0, 2.0, 100)
    if len(means.shape) == 0:
        means = means[None, ...]
        stds = stds[None, ...]
    for i in range(means.shape[0]):
        y = ch.exp(-0.5 * ((x - means[i]) / stds[i]) ** 2) / (stds[i] * (2 * 3.1415) ** 0.5)
        ax.plot(x, y, label=f"Component {i}")
    ax.set_title(title)
    ax.legend()

def plot_1d_gmm(means, stds, weights, ax, title=""):
    x = ch.linspace(-2.0, 2.0, 100)
    y = ch.zeros_like(x)
    if len(means.shape) == 0:
        means = means[None, ...]
        stds = stds[None, ...]
        weights = weights[None, ...]
    for i in range(means.shape[0]):
        y += weights[i] * ch.exp(-0.5 * ((x - means[i]) / stds[i]) ** 2) / (stds[i] * (2 * 3.1415) ** 0.5)
    ax.plot(x, y, label=f"GMM", linestyle='dashed', zorder=-1, linewidth=10.0)
    ax.set_title(title)
    ax.legend()

class ToyTask2DManager(BaseManager):

    def __init__(self, n_component, seed, device, **kwargs):
        super().__init__(seed, device, **kwargs)
        self.score_function = GMMScoreFunction.generate_random_params(4, 2)
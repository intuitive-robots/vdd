import torch
import torch.nn as nn

from vdd.networks.mlp import ResidualMLPNetwork


class InferenceNet(nn.Module):
    def __init__(self,
                 obs_dim,
                 n_components,
                 num_hidden_layer,
                 hidden_dim,
                 device='cuda'):

        super(InferenceNet, self).__init__()

        self.n_components = n_components

        self.network = ResidualMLPNetwork(obs_dim, n_components, hidden_dim, num_hidden_layer,
                                          dropout=0., device=device)

        self.trained = False

    def forward(self, observation):
        observation = self.network(observation)
        return torch.nn.functional.log_softmax(observation[..., :self.n_components], dim=-1)

    def sample(self, contexts):
        p = self.probabilities(contexts)
        thresholds = torch.cumsum(p, dim=-1)
        thresholds[:, -1] = 1.0
        eps = torch.rand(size=[contexts.shape[0], 1])
        samples = torch.argmax((eps < thresholds) * 1., dim=-1)
        return samples

    def probabilities(self, contexts):
        return torch.exp(self(contexts))

    def log_probabilities(self, contexts):
        return self(contexts)

    def entropies(self, contexts):
        p = self.probabilities(contexts)
        return -torch.sum(p * torch.log(p + 1e-25), dim=-1)

    def expected_entropy(self, contexts):
        return torch.mean(self.entropies(contexts))

    def kls(self, contexts, other):
        p = self.probabilities(contexts)
        other_log_p = other.log_probabilities(contexts)
        return torch.sum(p * (torch.log(p + 1e-25) - other_log_p), dim=-1)

    def expected_kl(self, contexts, other):
        return torch.mean(self.kls(contexts, other))

    def check_trained(self):
        if self.trained:
            return True
        else:
            raise ValueError('Inference network is not trained.')

    @property
    def params(self):
        return list(self.parameters())

    @property
    def param_norm(self):
        """
        Calculates the norm of network parameters.
        """
        return torch.norm(torch.stack([torch.norm(p.detach()) for p in self.parameters()]))

    @property
    def grad_norm(self):
        """
        Calculates the norm of current gradients.
        """
        return torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in self.parameters()]))

    def add_component(self):
        self.mask[self.n_components] = 1
        self.n_components += 1

    def to_gpu(self):
        self.to(torch.device('cuda'))

    def to_cpu(self):
        self.to(torch.device('cpu'))


class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, pred_log_resp, resp):
        return -(resp * pred_log_resp).mean()
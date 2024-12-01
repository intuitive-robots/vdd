from collections import deque
import torch
import torch.distributions as D
from torch import nn

import einops

from vdd.networks.gaussian import get_gmm_head
from vdd.networks.gating import GatingNet
from vdd.networks.mlp import ResidualMLPNetwork
from vdd.networks.network_utils import str2torchdtype


class GaussianMoE(nn.Module):
    def __init__(self, num_components, obs_dim, act_dim, prior_type, cmp_init, cmp_cov_type='diag',
                 backbone = None,
                 backbone_out_dim = 2,
                 cmp_mean_hidden_dims = 64,
                 cmp_mean_hidden_layers = 2,
                 cmp_cov_hidden_dims = 64,
                 cmp_cov_hidden_layers = 2,
                 cmp_activation="tanh",
                 cmp_init_std=1., cmp_minimal_std=1e-5,
                 learn_gating=False, gating_hidden_layers=4, gating_hidden_dims = 64,
                 dtype="float32", device="cpu", **kwargs):
        super(GaussianMoE, self).__init__()
        self.n_components = num_components
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.device = device
        self.dtype = str2torchdtype(dtype)

        self.backbone = backbone
        self.gmm_head = get_gmm_head(act_dim, num_components, cmp_init_std, cmp_minimal_std, cmp_cov_type, device=device)
        self.gmm_mean_net = ResidualMLPNetwork(input_dim=cmp_mean_hidden_dims,
                                               output_dim=self.gmm_head.flat_mean_dim,
                                               hidden_dim=cmp_mean_hidden_dims,
                                               num_hidden_layers=cmp_mean_hidden_layers,
                                               activation=cmp_activation,
                                               device=device)
        self.gmm_cov_net = ResidualMLPNetwork(input_dim=cmp_mean_hidden_dims,
                                              output_dim=self.gmm_head.flat_chol_dim,
                                              hidden_dim=cmp_cov_hidden_dims,
                                              num_hidden_layers=cmp_cov_hidden_layers,
                                              activation=cmp_activation,
                                              device=device)

        if hasattr(self.backbone, 'window_size'):
            self.window_size = self.backbone.window_size
        else:
            self.window_size = 1

        self.obs_contexts = deque(maxlen=self.window_size)

        self.learn_gating = learn_gating

        self.gating_network = GatingNet(self.joint_cmps._gpt.out_dim, num_components, gating_hidden_layers,
                                        gating_hidden_dims, device=device) if learn_gating else None

        if prior_type == 'uniform':
            self._prior = torch.ones(num_components, device=self.device, dtype=self.dtype) / num_components
        else:
            raise NotImplementedError(f"Prior type {prior_type} not implemented.")

    def reset(self):
        self.obs_contexts.clear()

    def forward(self, states, goals=None, train=True):
        self.train(train)

        if self.backbone is not None:
            x = self.backbone(states=states, goals=goals)
        else:
            x = torch.cat([states, goals], dim=-1) if goals is not None else states

        pre_means = self.gmm_mean_net(x)
        pre_chols = self.gmm_cov_net(x)
        cmp_means, cmp_chols = self.gmm_head(pre_means, pre_chols)
        cmp_means = einops.rearrange(cmp_means, 'b t c d -> b c t d')
        cmp_chols = einops.rearrange(cmp_chols, 'b t c d1 d2 -> b c t d1 d2')

        if self.gating_network is None:
            gating_probs = einops.repeat(self._prior, 'c -> b c t', b=states.shape[0], t=states.shape[1])
        else:
            x = self.joint_cmps._gpt(states, goals).detach()
            gating_probs = self.gating_network(x).exp() + 1e-8
            gating_probs = einops.repeat(gating_probs, 'b t c -> b c t')

        return cmp_means, cmp_chols, gating_probs

    def sample(self, cmp_means, cmp_chols, gating=None, n=1):
        if gating is None:
            prior = self._prior.unsqueeze(0).repeat(cmp_means.shape[0], 1)
            gating = D.Categorical(probs=prior)
        else:
            gating = D.Categorical(gating)

        return self.gmm_head.gmm_sample(cmp_means, cmp_chols, gating, n)


    @torch.no_grad()
    def act(self, state, goal=None, vision_task=False):
        if vision_task:
            self.agentview_image_contexts.append(state[0])
            self.inhand_image_contexts.append(state[1])
            self.robot_ee_pos_contexts.append(state[2])
            agentview_image_seq = torch.stack(list(self.agentview_image_contexts), dim=1)
            inhand_image_seq = torch.stack(list(self.inhand_image_contexts), dim=1)
            robot_ee_pos_seq = torch.stack(list(self.robot_ee_pos_contexts), dim=1)
            input_states = (agentview_image_seq, inhand_image_seq, robot_ee_pos_seq)
        else:
            self.obs_contexts.append(state)
            input_states = torch.stack(list(self.obs_contexts), dim=1)

        if goal is not None and len(goal.size()) == 2:
            goal = goal.unsqueeze(0)

        cmp_means, cmp_chols, gating = self(input_states, goal, train=False)

        cmp_means = cmp_means[..., -1, :].squeeze(0)
        gating = gating[..., -1].squeeze(0)

        if self.greedy_predict:
            indexs = gating.argmax(-1)
        else:
            gating_dist = D.Categorical(gating)
            indexs = gating_dist.sample([1])
        action_means = cmp_means[indexs, :]

        return action_means

    def get_parameter(self, target: str) -> "Parameter":
        if target == "gating":
            return self.gating_network.parameters()
        elif target == "cmps":
            return list(self.gmm_mean_net.parameters()) + list(self.gmm_cov_net.parameters()) + list(self.gmm_head.parameters())
        else:
            raise ValueError(f"Unknown target {target}")
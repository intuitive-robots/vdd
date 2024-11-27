from collections import deque
from typing import Dict

import torch
import torch as ch
import torch.distributions as D
from torch import nn

import einops

from common.models.policy_factory import get_policy_network
from common.utils.torch_utils import str2torchdtype
from common.utils.plot_utils import plot_2d_gaussians
from common.models.vision_encoders.vision_encoders_factory import get_visual_encoder


from vi.models.inference_net import InferenceNet

import matplotlib.pyplot as plt



class JointGaussianMixtureModel(nn.Module):
    def __init__(self, num_components, obs_dim, act_dim, prior_type, cmp_init, cmp_cov_type='diag',
                 cmp_hidden_dims = 64,
                 cmp_hidden_layers = 2,
                 cmp_cov_hidden_dims = 64,
                 cmp_cov_hidden_layers = 2,
                 greedy_predict=False,
                 share_layers=False, cmp_activation="tanh", cmp_contextual_std=True,
                 cmp_init_std=1., cmp_minimal_std=1e-5, learn_gating=False, gating_hidden_layers=4, gating_hidden_dims = 64,
                 vision_task=None, vision_encoder_params={},
                 dtype="float32", device="cpu", **kwargs):
        super(JointGaussianMixtureModel, self).__init__()
        self.n_components = num_components
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.device = device
        self.dtype = str2torchdtype(dtype)

        self.output_size = act_dim * self.n_components

        cmp_hidden_sizes = [cmp_hidden_dims] * cmp_hidden_layers
        cmp_cov_hidden_sizes = [cmp_cov_hidden_dims] * cmp_cov_hidden_layers

        self.train_vision_encoder = kwargs.get('train_vision_encoder', False)

        self.joint_cmps = get_policy_network(obs_dim=obs_dim, action_dim=act_dim, proj_type='bc', init=cmp_init,
                                             hidden_sizes=cmp_hidden_sizes,
                                             std_hidden_sizes=cmp_cov_hidden_sizes, activation=cmp_activation,
                                             cov_type=cmp_cov_type,
                                             contextual_std=cmp_contextual_std, init_std=cmp_init_std,
                                             minimal_std=cmp_minimal_std,
                                             share_layers=share_layers,
                                             n_components=num_components,
                                             **kwargs)
        if hasattr(self.joint_cmps, 'window_size'):
            self.window_size = self.joint_cmps.window_size
        else:
            self.window_size = 1

        self.obs_contexts = deque(maxlen=self.window_size)

        self.learn_gating = learn_gating

        self.greedy_predict = greedy_predict

        if learn_gating:
            if cmp_cov_type == 'transformer_gmm_full' or cmp_cov_type == 'transformer_gmm_diag':
                self.gating_network = InferenceNet(self.joint_cmps._gpt.out_dim, num_components, gating_hidden_layers, gating_hidden_dims, device=device)
            else:
                self.gating_network = InferenceNet(obs_dim, num_components, gating_hidden_layers, gating_hidden_dims,
                                                   device=device)
        else:
            self.gating_network = None

        if vision_task:
            self.vision_encoder = get_visual_encoder(vision_encoder_params).to(self.device, self.dtype)
            self.is_vision_task = True
            self.agentview_image_contexts = deque(maxlen=self.window_size)
            self.inhand_image_contexts = deque(maxlen=self.window_size)
            self.robot_ee_pos_contexts = deque(maxlen=self.window_size)
        else:
            self.vision_encoder = None
            self.is_vision_task = False

        self.log_gating = None

        self.joint_cmps = self.joint_cmps.to(self.device, self.dtype)

        self.cmp_cov_type = cmp_cov_type

        if prior_type == 'uniform':
            self._prior = ch.ones(num_components, device=self.device, dtype=self.dtype) / num_components
        else:
            raise NotImplementedError(f"Prior type {prior_type} not implemented.")

    def reset(self):
        self.obs_contexts.clear()
        if self.is_vision_task:
            self.agentview_image_contexts.clear()
            self.inhand_image_contexts.clear()
            self.robot_ee_pos_contexts.clear()


    def forward(self, states, goals=None, train=True):
        self.train(train)

        if self.vision_encoder is not None:
            b, t = states[0].shape[:2]
            states_dict = {"agentview_image": einops.rearrange(states[0], 'b t c h w -> (b t) c h w'),
                           "in_hand_image": einops.rearrange(states[1], 'b t c h w -> (b t) c h w'),
                           "robot_ee_pos": einops.rearrange(states[2], 'b t d -> (b t) d')}

            ###FIXME: freeze the vision encoder
            states = self.vision_encoder(states_dict)
            if not self.train_vision_encoder:
                states = states.detach()
            ###### reshape states
            states = einops.rearrange(states, '(b t) d -> b t d', b=b, t=t)

        cmp_means, cmp_chols = self.joint_cmps(states, goals, train=train)

        if self.gating_network is None:
            # gating_probs = self._prior.expand(cmp_means.shape[:-2] + self._prior.shape)
            gating_probs = einops.repeat(self._prior, 'c -> b c t', b=states.shape[0], t=cmp_means.shape[-2])
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

        comps = D.MultivariateNormal(cmp_means, scale_tril=cmp_chols, validate_args=False)
        gmm = D.MixtureSameFamily(gating, comps)
        return gmm.sample((n,))

    @ch.no_grad()
    def visualize_cmps(self, x):
        cmp_means, cmp_chols, gating = self(x, train=False)
        cmp_means = cmp_means.cpu()
        cmp_chols = cmp_chols.cpu()
        fig, ax = plt.subplots(1, 1)
        plot_2d_gaussians(cmp_means.squeeze(0), cmp_chols.squeeze(0), ax, title="GMM Components")
        ax.set_aspect('equal')
        plt.show()

    @ch.no_grad()
    def act(self, state, goal=None, vision_task=False):
        if vision_task:
            self.agentview_image_contexts.append(state[0])
            self.inhand_image_contexts.append(state[1])
            self.robot_ee_pos_contexts.append(state[2])
            agentview_image_seq = ch.stack(list(self.agentview_image_contexts), dim=1)
            inhand_image_seq = ch.stack(list(self.inhand_image_contexts), dim=1)
            robot_ee_pos_seq = ch.stack(list(self.robot_ee_pos_contexts), dim=1)
            input_states = (agentview_image_seq, inhand_image_seq, robot_ee_pos_seq)
        else:
            self.obs_contexts.append(state)
            input_states = ch.stack(list(self.obs_contexts), dim=1)

        # if len(input_states.size()) == 2:
        #     input_states = input_states.unsqueeze(0)
        if goal is not None and len(goal.size()) == 2:
            goal = goal.unsqueeze(0)

        cmp_means, cmp_chols, gating = self(input_states, goal, train=False)

        ### only use the last time step for prediction
        cmp_means = cmp_means[..., -1, :].squeeze(0)
        gating = gating[..., -1].squeeze(0)

        if self.greedy_predict:
            indexs = gating.argmax(-1)
        else:
            gating_dist = D.Categorical(gating)
            indexs = gating_dist.sample([1])
        action_means = cmp_means[indexs, :]

        return action_means

    def log_responsibilities(self, pred_means, pred_chols, pred_gatings, samples):
        """
        b -- state batch
        c -- the number of components
        v -- the number of vi samples
        a -- action dimension
        o -- observation dimension
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
        log_probs_cmps = self.joint_cmps.log_probability((pred_means, pred_chols), samples)

        ### log_probs: (b, c, v)
        log_probs = log_probs_cmps.clone()
        log_probs = torch.einsum('ijj...->ij...', log_probs)

        if self.learn_gating:
            log_gating = ch.log(pred_gatings)
        else:
            log_gating = ch.log(self._prior).view(1, -1)

        probs_cmps = log_probs_cmps.exp()

        ### Do we need to detach the log_margin?
        if self.learn_gating:
            margin = ch.einsum('ijkl,ik->ijl', probs_cmps, pred_gatings)
            log_margin = ch.log(margin + 1e-8)
        else:
            log_margin = ch.log(ch.einsum('ijkl,k->ijl', probs_cmps, self._prior))

        return log_probs + log_gating.unsqueeze(-1) - log_margin
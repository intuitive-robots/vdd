from vdd.score_functions.score_base import ScoreFunction

from agents.ddpm_agent import DiffusionAgent

from copy import deepcopy

import torch
import einops

class DDPMScoreFunction(ScoreFunction):
    def __init__(self, model: DiffusionAgent, sigma_index=-1, obs_dim=10, goal_dim=10, weights_type='srpo',
                 t_min=1, t_max=8, t_bound=8, anneal_end_iter=1e6,
                 noise_level_type='uniform', device='cuda', **kwargs):
        super().__init__(model)
        self.sigma_index = sigma_index
        self.goal_dim = goal_dim
        self.obs_dim = obs_dim
        self.weights_type = weights_type
        self.t_min = t_min
        self.t_max = t_max
        self.t_bound = t_bound
        self.annealing_end_iter = anneal_end_iter
        self.noise_level_type = noise_level_type
        self.device = device

        self.vision_task = kwargs.get('vision_task', False)

        if self.vision_task:
            self.noise_network = self.model.model.model.model
            self.betas = self.model.model.model.betas.clone()
            self.sqrt_one_minus_alphas_cumprod = self.model.model.model.sqrt_one_minus_alphas_cumprod.clone()
        else:
            self.noise_network = self.model.model.model
            self.betas = self.model.model.betas.clone()
            self.sqrt_one_minus_alphas_cumprod = self.model.model.sqrt_one_minus_alphas_cumprod.clone()

        print("DDPM Score Function Initialized")

    def __call__(self, samples, states, goals=None, iter=None, vision_task=False):
        return self._get_score(samples, states, goals, iter, vision_task)

    @torch.no_grad()
    def _get_score(self, samples, state, goal, iter=None, vision_task=False):
        self.noise_network.eval()

        noise_level = self._get_noise_level(samples, noise_level_type=self.noise_level_type, iter=iter).to(self.device)

        weights = self._get_weights(noise_level[..., None, None], weights_type=self.weights_type).to(self.device)

        ### einpack the samples
        # b = samples.shape[0]
        # c = samples.shape[1]
        # v = samples.shape[2]

        (b, c, v, t) = samples.shape[:4]

        if vision_task:
            # self.model.model.obs_encoder.eval()
            ### hack for vision-based tasks
            agent_view_image = einops.rearrange(state[0], 'b t ... -> (b t) ... ')
            in_hand_image = einops.rearrange(state[1], 'b t ... -> (b t) ... ')
            robot_ee_pos = einops.rearrange(state[2], 'b t ... -> (b t) ... ')
            state_dict = {"agentview_image": agent_view_image,
                          "in_hand_image": in_hand_image,
                          "robot_ee_pos": robot_ee_pos}
            try:
                state = self.model.model.obs_encoder(state_dict)
            except Exception as e:
                print("error: ", e)
                print("Error in encoding the state")

            pack_state = einops.rearrange(state, '(b t) ... -> b t ...', b=b, t=t)
            pack_state = einops.repeat(pack_state, 'b t ... -> b c v t ...', c=c, v=v)
            pack_state = einops.rearrange(pack_state, 'b c v t ... -> (b c v) t ...')
        else:
            pack_state = einops.rearrange(state, 'b c v t ... -> (b c v) t ...')
        pack_samples = einops.rearrange(samples, 'b c v t ... -> (b c v) t ...')
        pack_goal = einops.rearrange(goal, 'b c v t ... -> (b c v) t ...') if goal is not None else None
        pack_noise_level = einops.rearrange(noise_level, 'b c v -> (b c v)')

        noise = self.noise_network(actions=pack_samples, time=pack_noise_level, states=pack_state, goals=pack_goal)

        ### unpack the denoised samples
        noise = einops.rearrange(noise, '(b c v) t d -> b c v t d', b=b, c=c, v=v)

        ### score epsilon(x;sigma) / sqrt(beta)
        ###TODO:FIXME: check if the broadcast is correct
        if self.noise_level_type == 'discrete':
            # score = - noise / torch.sqrt((self.betas[noise_level]))[..., None, None]
            score = - noise / self.sqrt_one_minus_alphas_cumprod[noise_level][..., None, None]
        elif self.noise_level_type == 'uniform':
            score = - noise / ((noise_level[..., None, None].float() + 1e-4)/self.t_bound)
        else:
            raise ValueError(f"Unknown noise level type: {self.noise_level_type}, expected 'discrete' or 'uniform'")

        return score * weights, noise_level


    def _get_noise_level(self, samples, noise_level_type='uniform', iter=None):

        if noise_level_type == 'discrete':
            # torch.randint is exclusive of the upper bound
            sampled_t = torch.randint(self.t_min, self.t_max, samples.shape[:3])
        elif noise_level_type == 'uniform':
            sampled_t = torch.rand(samples.shape[:3]) * (self.t_max - self.t_min) + self.t_min
        else:
            raise ValueError(f"Unknown noise level type: {noise_level_type}, expected 'discrete' or 'uniform'")
        return sampled_t

    def _get_weights(self, noise_level, weights_type='stable'):
        if weights_type == 'stable':
            return torch.ones_like(noise_level)
        else:
            raise ValueError(f"Unknown weights type: {weights_type}")
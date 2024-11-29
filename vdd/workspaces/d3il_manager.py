import os
import torch as ch
import numpy as np
import einops
from collections import deque

import hydra
from omegaconf import OmegaConf

import torch.utils.data as Data

from vi.experiment_managers.base_manager import BaseManager
from vi.score_functions.beso_score import BesoScoreFunction
from vi.score_functions.ddpm_score import DDPMScoreFunction

from matplotlib import pyplot as plt

class D3ILAgent:
    def __init__(self, gmm_agent, scaler):
        self.gmm_agent = gmm_agent
        self.scaler = scaler

    def predict(self, obs, if_vision=False):

        if if_vision:
            agent_view_image = ch.from_numpy(obs[0]).to(self.gmm_agent.device).to(ch.float32).unsqueeze(0)
            in_hand_image = ch.from_numpy(obs[1]).to(self.gmm_agent.device).to(ch.float32).unsqueeze(0)
            robot_ee_pos = self.scaler.scale_input(ch.from_numpy(obs[2]).to(self.gmm_agent.device)).to(ch.float32).unsqueeze(0)
            obs = (agent_view_image, in_hand_image, robot_ee_pos)
            act = self.gmm_agent.act(obs, vision_task=True)
            act = self.scaler.inverse_scale_output(act)
            return act.cpu().numpy()

        obs = ch.from_numpy(obs).unsqueeze(0).to(self.gmm_agent.device).to(ch.float32)
        obs = self.scaler.scale_input(obs)
        act = self.gmm_agent.act(obs)
        act = self.scaler.inverse_scale_output(act)
        return act.cpu().numpy()

    def reset(self):
        self.gmm_agent.reset()

class D3ILManager(BaseManager):
    def __init__(self, model_path, sv_name, seed=0, device='cuda', score_fn_params=None, vision_task=False,
                 goal_conditioned=False, score_type='beso', **kwargs):
        super().__init__(seed, device, **kwargs)
        datasets_config = kwargs.get("datasets_config", None)
        self.agent, self.env_sim = self.get_agent_and_workspace(model_path, sv_name, datasets_config)
        self.scaler = self.agent.scaler
        if score_fn_params is not None:
            if score_type == 'beso':
                self.score_function = BesoScoreFunction(self.agent, **score_fn_params)
            elif score_type == 'ddpm':
                self.score_function = DDPMScoreFunction(self.agent, **score_fn_params)
            else:
                raise NotImplementedError
        else:
            self.score_function = None
        self.cpu_cores = kwargs.get("cpu_cores", None)
        self.is_vision_task = vision_task
        self.goal_conditioned = goal_conditioned

    def env_rollout(self, agent, n_episodes: int, render=False, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories = n_episodes
        self.env_sim.render = render
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict

    def get_scaler(self, **kwargs):
        return self.scaler

    def preprocess_dataloader(self, dataloader, **kwargs):
        obs = []
        actions = []
        for batch in dataloader:
            obs.append(batch[0])
            actions.append(batch[1])
        obs = ch.cat(obs, dim=0).squeeze(1).to(self.device)
        obs = self.scaler.scale_input(obs)
        actions = ch.cat(actions, dim=0).squeeze(1).to(self.device)
        actions = self.scaler.scale_output(actions)
        idx = ch.arange(obs.shape[0]).long().to(self.device)
        return Data.TensorDataset(idx, obs, actions)

    def get_score_function(self, **kwargs):
        return self.score_function

    def get_train_and_test_datasets(self, **kwargs):
        #TODO: Hackey way to get train and test datasets
        return self.agent.train_dataloader, self.agent.test_dataloader

    def plot_grad_field(self, state):

        state_0, state_1, state_2 = state

        raw_x, raw_y = np.meshgrid(np.linspace(-2, 2, 10), np.linspace(-2, 2, 10))

        x = ch.tensor(raw_x).to(self.device).to(ch.float32).unsqueeze(-1)
        y = ch.tensor(raw_y).to(self.device).to(ch.float32).unsqueeze(-1)

        grid_actions = ch.cat([x, y], axis=-1).to(self.device).to(ch.float32).reshape(-1, 2)

        t = 5

        grid_actions = einops.rearrange(grid_actions, 'n d -> 1 1 n d')

        grid_actions = einops.repeat(grid_actions, '1 1 n d -> 1 1 n t d', t=t)

        state_0 = einops.rearrange(state_0, '... -> 1 ...')
        state_1 = einops.rearrange(state_1, '... -> 1 ...')
        state_2 = einops.rearrange(state_2, '... -> 1 ...')

        state = (state_0, state_1, state_2)

        score = self.score_function(samples=grid_actions, states=state, vision_task=True)

        score = score[0].squeeze()
        score = score[:, -1, :]

        score = score.reshape(10, 10, 2).cpu().numpy()

        u = score[..., 0]
        v = score[..., 1]

        plt.quiver(raw_x, raw_y, u, v)

    def preprocess_data(self, batch_data):
        if self.is_vision_task:
            obs_batch = batch_data[:3]
            agentview_image = obs_batch[0].to(self.device)
            in_hand_image = obs_batch[1].to(self.device)
            robot_ee_pos = self.scaler.scale_input(obs_batch[2].to(self.device))

            obs_batch = (agentview_image, in_hand_image, robot_ee_pos)

            action_batch = self.scaler.scale_output(batch_data[3]).to(self.device)

            # for idx in range(10):
            #     self.plot_grad_field((obs_batch[0][idx], obs_batch[1][idx], obs_batch[2][idx]))
            #     normalized_action = action_batch[idx, -1, :].cpu().numpy()
            #     plt.plot(normalized_action[0], normalized_action[1], 'ro')
            #     plt.show()

            # print("finished")

            if self.goal_conditioned:
                return obs_batch, action_batch, batch_data[4]
            else:
                return obs_batch, action_batch, None
        else:
            obs_batch = batch_data[0]
            action_batch = batch_data[1]
            scaled_obs = self.scaler.scale_input(obs_batch).to(self.device)
            scaled_action = self.scaler.scale_output(action_batch).to(self.device)
            return scaled_obs, scaled_action, None
    def preprocess_config(self, config):
        return config

    def get_agent_and_workspace(self, model_path, sv_name, datasets_config=None):
        cfg_store_path = os.path.join(model_path, ".hydra", "config.yaml")
        config = OmegaConf.load(cfg_store_path)

        config = self.preprocess_config(config)

        np.random.seed(self.seed)
        ch.manual_seed(self.seed)

        if datasets_config is not None:
            assert datasets_config['window_size'] == config.agents.window_size, "Window size mismatch"
            if 'goal_seq_len' in datasets_config:
                assert datasets_config['goal_seq_len'] == config.agents.goal_window_size, "Goal sequence length mismatch"
            config.agents.num_workers = datasets_config['num_workers']
            config.agents.train_batch_size = datasets_config['train_batch_size']
            config.agents.val_batch_size = datasets_config['test_batch_size']

        agent = hydra.utils.instantiate(config.agents)
        agent.load_pretrained_model(model_path, sv_name=sv_name)
        env_sim = hydra.utils.instantiate(config.simulation)
        return agent, env_sim


class D3ILAlignManager(D3ILManager):
    def env_rollout(self, agent, n_episodes: int, num_ctxts: int = 10, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories_per_context = n_episodes
        self.env_sim.n_contexts = num_ctxts
        self.env_sim.render = False
        if self.cpu_cores is not None and len(self.cpu_cores) > 20:
            self.cpu_cores = self.cpu_cores[:20]
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict

    def preprocess_config(self, config):
        config['trainset']['_target_'] = 'environments.dataset.aligning_dataset.Aligning_Dataset'
        config['valset']['_target_'] = 'environments.dataset.aligning_dataset.Aligning_Dataset'
        config['simulation']['_target_'] = 'simulation.aligning_sim.Aligning_Sim'
        config['train_data_path'] = 'environments/dataset/data/aligning/train_files.pkl'
        config['eval_data_path'] = 'environments/dataset/data/aligning/eval_files.pkl'
        return config

class D3ILSortingVisionManager(D3ILManager):
    def env_rollout(self, agent, n_episodes: int, num_ctxts:int=10, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories_per_context = n_episodes
        self.env_sim.n_contexts = num_ctxts
        self.env_sim.render = False
        ## Limiting the number of cores to 8, otherwise it will run out of memory even on Horeka
        if self.cpu_cores is not None and len(self.cpu_cores) > 10:
            self.cpu_cores = self.cpu_cores[:10]
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict

    def preprocess_config(self, config):
        return config

class D3ILStackingManager(D3ILManager):
    def env_rollout(self, agent, n_episodes: int, num_ctxts:int=10, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories_per_context = n_episodes
        self.env_sim.n_contexts = num_ctxts
        self.env_sim.render = False
        if self.cpu_cores is not None and len(self.cpu_cores) > 20:
            self.cpu_cores = self.cpu_cores[:20]
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict
    def preprocess_config(self, config):
        config['trainset']['_target_'] = 'environments.dataset.stacking_dataset.Stacking_Dataset'
        config['valset']['_target_'] = 'environments.dataset.stacking_dataset.Stacking_Dataset'
        config['simulation']['_target_'] = 'simulation.stacking_sim.Stacking_Sim'
        config['train_data_path'] = 'environments/dataset/data/stacking/train_files.pkl'
        config['eval_data_path'] = 'environments/dataset/data/stacking/eval_files.pkl'
        return config


class D3ILStackingVisionManager(D3ILManager):
    def env_rollout(self, agent, n_episodes: int, num_ctxts:int=10, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories_per_context = n_episodes
        self.env_sim.n_contexts = num_ctxts
        self.env_sim.render = False
        ## Limiting the number of cores to 8, otherwise it will run out of memory even on Horeka
        if self.cpu_cores is not None and len(self.cpu_cores) > 10:
            self.cpu_cores = self.cpu_cores[:10]
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict

    def preprocess_config(self, config):
        config['trainset']['_target_'] = 'environments.dataset.stacking_dataset.Stacking_Img_Dataset'
        config['valset']['_target_'] = 'environments.dataset.stacking_dataset.Stacking_Img_Dataset'
        config['simulation']['_target_'] = 'simulation.stacking_vision_sim.Stacking_Sim'
        config['train_data_path'] = 'environments/dataset/data/stacking/vision_train_files.pkl'
        config['eval_data_path'] = 'environments/dataset/data/stacking/vision_eval_files.pkl'
        return config

class D3ILAvoidingManager(D3ILManager):
    def env_rollout(self, agent, n_episodes: int, render=False, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories = n_episodes
        self.env_sim.render = render
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict

    def preprocess_config(self, config):
        config['trainset']['_target_'] = 'environments.dataset.avoiding_dataset.Avoiding_Dataset'
        config['valset']['_target_'] = 'environments.dataset.avoiding_dataset.Avoiding_Dataset'
        config['simulation']['_target_'] = 'simulation.avoiding_sim.Avoiding_Sim'
        config['data_directory'] = 'environments/dataset/data/avoiding/data'
        return config

class D3ILPushingManager(D3ILManager):
    def env_rollout(self, agent, n_episodes: int, num_ctxts:int=10, **kwargs):
        agent.eval()
        ch.cuda.empty_cache()
        d3il_agent = D3ILAgent(agent, self.scaler)
        self.env_sim.n_trajectories_per_context = n_episodes
        self.env_sim.n_contexts = num_ctxts
        self.env_sim.render = False
        if self.cpu_cores is not None and len(self.cpu_cores) > 20:
            self.cpu_cores = self.cpu_cores[:20]
        eval_dict = self.env_sim.test_agent(d3il_agent, self.cpu_cores)
        print(eval_dict)
        agent.train()
        return eval_dict
    def preprocess_config(self, config):
        config['trainset']['_target_'] = 'environments.dataset.pushing_dataset.Pushing_Dataset'
        config['valset']['_target_'] = 'environments.dataset.pushing_dataset.Pushing_Dataset'
        config['simulation']['_target_'] = 'simulation.pushing_sim.Pushing_Sim'
        config['train_data_path'] = 'environments/dataset/data/pushing/train_files.pkl'
        config['eval_data_path'] = 'environments/dataset/data/pushing/eval_files.pkl'
        return config


if __name__ == "__main__":
    # config_dir = "/home/hongyi/Codes/demo_acc_rl/d3il/configs"
    # config_name = "avoiding_config.yaml"
    model_path = "/home/hongyi/Codes/demo_acc_rl/d3il/logs/models/stacking/beso/agent_name=beso,agents.num_sampling_steps=16,agents.sigma_max=1,agents.sigma_min=0.01,agents=beso_agent,group=beso_train_loss,seed=0,simulation.n_contexts=60,simulation.n_cores=30,simulation.n_trajectories_per_context=18,window_size=5"
    sv_name = "eval_best_beso.pth"
    manager = D3ILManager(model_path, sv_name,
                          {'noise_level_type': "last_sigma",
                           'weights_type': "stable"},
                          obs_dim=4, goal_dim=0,
                          seed=0, device="cuda")
    agent, env_sim = manager.agent, manager.env_sim
    manager.env_sim.render = True
    manager.env_sim.n_trajectories_per_context = 4
    manager.env_sim.n_contexts = 2
    success, entropy, mean_dist = manager.env_sim.test_agent(manager.agent, [1])
    print(success, entropy, mean_dist)
    # manager.env_rollout(manager.agent, 10)
    train_dataset, test_dataset = manager.get_train_and_test_datasets()
    manager.get_scaler()

    # agent, env_sim = manager.get_agent_and_workspace(model_path, sv_name)

import os
import torch as ch
import torch.utils.data as Data
import numpy as np

from collections import deque

from vi.experiment_managers.base_manager import BaseManager
from vi.score_functions.beso_score import BesoScoreFunction

import hydra
from omegaconf import OmegaConf

from copy import deepcopy

import gym
import adept_envs

class KitchenAgentWrapper:
    def __init__(self, gmm_agent, scaler):
        self.gmm_agent = gmm_agent
        self.scaler = scaler

    def predict(self, obs_dict, **kwargs):
        obs = obs_dict['observation']
        goal = obs_dict['goal_observation']
        obs = self.scaler.scale_input(obs)
        goal = self.scaler.scale_input(goal)
        act = self.gmm_agent.act(obs, goal)
        act = self.scaler.clip_action(act)
        act = self.scaler.inverse_scale_output(act)
        return act

    def reset(self):
        self.gmm_agent.reset()

class KitchenManager(BaseManager):

    def __init__(self, model_path, sv_name, score_fn_params, seed, device, **kwargs):
        super().__init__(seed, device, **kwargs)
        datasets_config = kwargs.get("datasets_config", None)
        self.agent, self.workspace_manager = self.get_agent_and_workspace(model_path, sv_name, datasets_config)
        self.score_function = BesoScoreFunction(self.agent, obs_dim=30, goal_dim=30, **score_fn_params)
        self.scaler = self.workspace_manager.scaler
        self.cpu_cores = kwargs.get("cpu_cores", None)
        self.goal_idx_offset = 0

    def env_rollout(self, agent, n_episodes: int, **kwargs):
        # agent.eval()
        #
        # env = gym.make(self.workspace_manager.env_name)
        # env.seed(self.seed)
        # rewards = []
        # if self.goal_idx_offset > 535:
        #     self.goal_idx_offset = 0
        # for goal_idx in range(n_episodes):
        #     goal_idx = goal_idx + self.goal_idx_offset
        #     total_reward = 0
        #     done = False
        #     obs = env.reset()
        #     goal = self.workspace_manager.multi_goals_fn(obs, goal_idx, 0)
        #     for n in range(280):
        #         if done or n == 279:
        #             rewards.append(total_reward)
        #             print(f" Goal index {goal_idx}, Total reward: {total_reward}")
        #             break
        #         obs = ch.from_numpy(obs).to(self.device).to(ch.float32)[..., :30]
        #         obs = obs.unsqueeze(0)
        #         # input = ch.cat([obs, goal], dim=-1).to(self.device).to(ch.float32)
        #         action = agent.act(obs, goal)
        #
        #         try:
        #             obs, reward, done, info = env.step(action.squeeze().cpu().numpy())
        #         except:
        #             print("unstable simulation")
        #             reward = 0
        #         total_reward += reward
        # self.goal_idx_offset += n_episodes
        # agent.train()
        # return {"mean_rewards": sum(rewards)/len(rewards)}
        if self.cpu_cores is not None:
            os.sched_setaffinity(os.getpid(), set([list(self.cpu_cores)[0]]))
        ch.cuda.empty_cache()
        agent.eval()
        wrapped_agent = KitchenAgentWrapper(agent, self.scaler)
        self.workspace_manager.eval_n_times = n_episodes
        return_dict, _ = self.workspace_manager.test_agent(wrapped_agent, log_wandb=False)
        agent.train()
        return return_dict

    def pre_process_dataset(self, dataset, keep_window=True):
        obs = []
        actions = []
        goals = []
        for i in range(len(dataset)):
            obs.append(dataset[i]['observation'])
            actions.append(dataset[i]['action'])
            goals.append(dataset[i]['goal_observation'])

        obs = ch.cat(obs, dim=0).to(self.device).to(ch.float32)
        actions = ch.cat(actions, dim=0).to(self.device).to(ch.float32)
        goals = ch.cat(goals, dim=0).to(self.device).to(ch.float32)
        idx = ch.arange(0, obs.shape[0]).long().to(self.device)
        inputs = ch.cat([obs, goals], dim=-1)
        return Data.TensorDataset(idx, inputs, actions)

    def get_train_and_test_datasets(self, **kwargs):
        return self.workspace_manager.data_loader['train'], self.workspace_manager.data_loader['test']

    def preprocess_data(self, batch_data):
        scaled_obs = self.scaler.scale_input(batch_data['observation'])
        scaled_output = self.scaler.scale_output(batch_data['action'])
        scaled_goal_obs = self.scaler.scale_input(batch_data['goal_observation'])
        return scaled_obs, scaled_output, scaled_goal_obs

    def get_scaler(self, **kwargs):
        return deepcopy(self.agent.scaler)

    def get_score_function(self, **kwargs):
        return self.score_function

    def get_agent_and_workspace(self, model_path, sv_name, datasets_config=None):
        cfg_store_path = os.path.join(model_path, ".hydra", "config.yaml")
        config = OmegaConf.load(cfg_store_path)
        np.random.seed(self.seed)
        ch.manual_seed(self.seed)
        agent = hydra.utils.instantiate(config.agents)
        agent.load_pretrained_model(model_path, sv_name=sv_name)

        if datasets_config is not None:
            assert datasets_config['window_size'] == config.workspaces.dataset_fn.window_size, "Window size mismatch"
            # config.workspaces.dataset_fn.window_size = datasets_config['window_size']
            config.workspaces.dataset_fn.train_fraction = datasets_config['train_fraction']
            config.workspaces.num_workers = datasets_config['num_workers']
            config.workspaces.train_batch_size = datasets_config['train_batch_size']
            config.workspaces.test_batch_size = datasets_config['test_batch_size']

        workspace_manager = hydra.utils.instantiate(config.workspaces)

        return agent, workspace_manager

if __name__=="__main__":
    config_dir = "/home/hongyi/Codes/score_rl/beso/configs"
    config_name = "evaluate_kitchen.yaml"
    score_fn_params = {"noise_level_type": "uniform", "weights_type": "stable"}
    seed = 0
    device = "cuda"

    kitchen_manager = KitchenManager(config_dir, config_name, score_fn_params, seed, device)
    print(kitchen_manager.get_train_and_test_datasets())
    print(kitchen_manager.get_scaler())
    print(kitchen_manager.get_score_function())
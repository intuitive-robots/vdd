import os
import torch
import torch as ch
import numpy as np

from vdd.workspaces.base_manager import BaseManager
from vdd.score_functions.beso_score import BesoScoreFunction

import hydra
from omegaconf import OmegaConf
def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)

class AgentWrapper:
    def __init__(self, gmm_agent, scaler):
        self.gmm_agent = gmm_agent
        self.scaler = scaler

    @ch.no_grad()
    def predict(self, obs_dict, **kwargs):
        obs = obs_dict['observation']
        goal = obs_dict['goal_observation']
        obs = self.scaler.scale_input(obs)
        goal = self.scaler.scale_input(goal)
        goal[..., [2, 5, 6, 7, 8, 9]] = 0
        act = self.gmm_agent.act(obs, goal)
        act = self.scaler.clip_action(act)
        act = self.scaler.inverse_scale_output(act)
        return act

    def reset(self):
        self.gmm_agent.reset()

class BlockPushManager(BaseManager):

        def __init__(self, model_path, sv_name, seed, device, score_fn_params=None, **kwargs):
            super().__init__(seed, device, **kwargs)
            datasets_config = kwargs.get("datasets_config", None)
            self.beso_agent, self.workspace_manager = self.get_agent_and_workspace(model_path, sv_name, datasets_config)
            if score_fn_params is not None:
                self.score_function = BesoScoreFunction(self.beso_agent, obs_dim=10, goal_dim=10, **score_fn_params)
            else:
                self.score_function = None
            self.scaler = self.workspace_manager.scaler
            self.seed = seed
            self.train_fraction = self.workspace_manager.train_fraction
            self.goal_idx_offset = 0
            self.push_traj = self.workspace_manager.push_traj
            self.cpu_cores = kwargs.get("cpu_cores", None)

        def env_rollout(self, agent, n_episodes: int, **kwargs):
            """
            evaluate the agent
            :return:
            """
            if self.cpu_cores is not None:
                assign_process_to_cpu(os.getpid(), set([int(list(self.cpu_cores)[0])]))
            ch.cuda.empty_cache()
            agent.eval()
            wrapped_agent = AgentWrapper(agent, self.scaler)
            self.workspace_manager.eval_n_times = n_episodes
            return_dict = self.workspace_manager.test_agent(wrapped_agent, log_wandb=False)
            agent.train()
            return return_dict

        def get_train_and_test_datasets(self, **kwargs):
            return self.workspace_manager.data_loader['train'], self.workspace_manager.data_loader['test']

        def preprocess_data(self, batch_data):
            scaled_obs = self.scaler.scale_input(batch_data['observation'])
            scaled_actions = self.scaler.scale_output(batch_data['action'])
            scaled_goals = self.scaler.scale_input(batch_data['goal_observation'])
            scaled_goals[..., [2, 5, 6, 7, 8, 9]] = 0
            return scaled_obs, scaled_actions, scaled_goals

        def get_scaler(self,):
            return self.workspace_manager.scaler

        def get_score_function(self, **kwargs):
            return self.score_function

        def get_agent_and_workspace(self, model_path, sv_name, datasets_config=None):
            cfg_store_path = os.path.join(model_path, ".hydra", "config.yaml")
            config = OmegaConf.load(cfg_store_path)
            np.random.seed(self.seed)
            ch.manual_seed(self.seed)
            config.seed = self.seed
            agent = hydra.utils.instantiate(config.agents)
            agent.load_pretrained_model(model_path, sv_name=sv_name)

            if datasets_config is not None:
                config.workspaces.seed = self.seed
                config.workspaces.dataset_fn.random_seed = self.seed
                config.workspaces.goal_fn.seed = self.seed
                assert datasets_config[
                           'window_size'] == config.workspaces.dataset_fn.window_size, "Window size mismatch"
                if 'goal_seq_len' in datasets_config:
                    assert datasets_config['goal_seq_len'] == config.workspaces.goal_fn.goal_seq_len, "Goal sequence length mismatch"
                config.workspaces.dataset_fn.train_fraction = datasets_config['train_fraction']
                config.workspaces.num_workers = datasets_config['num_workers']
                config.workspaces.train_batch_size = datasets_config['train_batch_size']
                config.workspaces.test_batch_size = datasets_config['test_batch_size']

            workspace_manager = hydra.utils.instantiate(config.workspaces)

            return agent, workspace_manager
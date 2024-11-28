import os
from typing import Dict

import abc
import copy

import torch
import torch as ch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


import einops

from common.utils.network_utils import get_lr_schedule, get_optimizer

from vi.models.joint_gmm_policy import JointGaussianMixtureModel
from vi.models.inference_net import SoftCrossEntropyLoss

# from beso.envs.block_pushing.block_pushing_multimodal import BlockPushMultimodal




class AmortizedVID_jointGMM(abc.ABC):
    def __init__(self,
                 agent: JointGaussianMixtureModel,
                 cmps_optimizer: ch.optim.Optimizer,
                 gating_optimizer: ch.optim.Optimizer,
                 score_function: BesoScoreFunction,
                 train_dataloader: DataLoader,
                 test_dataloader: DataLoader,
                 train_batch_size: int = 512,
                 vi_batch_size: int = 2,
                 data_shuffle: bool = True,
                 cmps_lr_scheduler = None,
                 gating_lr_scheduler = None,
                 cmp_steps: int = 1,
                 gating_steps: int = 1,
                 learn_gating: bool = True,
                 fix_gating_after_iters: int = 0,
                 max_train_iters: int = 10000,
                 detach_chol: bool = False,
                 seed: int = 0,
                 device: str = 'cuda',
                 dtype: str = 'float32',
                 **kwargs
                 ):
        self.agent = agent
        self.is_vision_task = self.agent.is_vision_task

        self.batch_size = train_batch_size
        self.data_shuffle = data_shuffle
        self.max_train_iters = max_train_iters
        self.goal_idx_offset = 0

        self.score_function = score_function

        self.cmps_optimizer = cmps_optimizer
        self.gating_optimizer = gating_optimizer
        self.cmps_lr_scheduler = cmps_lr_scheduler
        self.gating_lr_scheduler = gating_lr_scheduler

        self.learn_gating = learn_gating

        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader
        self.iter_train_dataloader = iter(train_dataloader)

        self.vi_batch_size = vi_batch_size
        self.scaler = None
        self.exp_manager = None

        self.detach_chol = detach_chol

        self.cmp_steps = cmp_steps
        self.gating_steps = gating_steps
        self.fix_gating_after_iters = fix_gating_after_iters

        self.train_dataset = train_dataloader

        self.device = device
        self.dtype = ch.float64 if dtype == 'float64' else ch.float32
        self.seed = seed

    def get_scaler(self, scaler):
        self.scaler = scaler

    def get_exp_manager(self, exp_manager):
        self.exp_manager = exp_manager

    def init_agent(self,):
        """
        update the normalizer and reset the agent
        :return:
        """
        pass

    def iterative_train(self, n: int):
        """
        train the agent
        :return:
        """
        train_metric_dict = {}

        for i in range(self.cmp_steps):
            try:
                batch = next(self.iter_train_dataloader)
            except StopIteration:
                self.iter_train_dataloader = iter(self.train_dataloader)
                batch = next(self.iter_train_dataloader)
            batch = self.exp_manager.preprocess_data(batch)
            train_metric_dict = self.iterative_train_cmp(batch, n)

        if self.learn_gating and n < self.fix_gating_after_iters:
            for i in range(self.gating_steps):
                try:
                    batch = next(self.iter_train_dataloader)
                except StopIteration:
                    self.iter_train_dataloader = iter(self.train_dataloader)
                    batch = next(self.iter_train_dataloader)
                batch = self.exp_manager.preprocess_data(batch)
                train_metric_dict.update(self.armotized_vi_update(batch[0], batch[1], batch[2]))

        if self.learn_gating and n == self.fix_gating_after_iters:
            self.agent.gating_network.train(mode=False)

        return train_metric_dict

    @ch.no_grad()
    def iterative_evaluate(self, ):
        """
        evaluate the agent
        :return:
        """
        ch.cuda.empty_cache()
        self.agent.eval()
        test_losses = []
        gatings = []

        logging_dict = {}
        for batch in self.test_dataloader:
            batch = self.exp_manager.preprocess_data(batch)
            inputs = batch[0]
            outputs = batch[1]
            if batch[2] is not None:
                goals = batch[2].to(self.device).to(self.dtype)
                goals = self.scaler.scale_input(goals)
            else:
                goals = None

            # inputs = self.scaler.scale_input(inputs)
            # outputs = self.scaler.scale_output(outputs)

            # equivalent to self.agent.act(inputs), here unfold to record the gating
            cmp_means, _, gating = self.agent(inputs, goals, train=False)

            cmp_means = einops.rearrange(cmp_means, 'b c t a -> (b t) c a')
            outputs = einops.rearrange(outputs, 'b t a -> (b t) a')

            gating = einops.rearrange(gating, 'b c t -> (b t) c')

            if torch.isnan(cmp_means).any():
                print("nan in cmp_means")
            if torch.isnan(gating).any():
                print("nan in gating_prediction")
            gating_dist = ch.distributions.Categorical(gating)

            batch_indices = ch.arange(0, cmp_means.size(0)).unsqueeze(-1)

            indices = gating_dist.sample([1]).swapaxes(0, 1)

            pred_outputs = cmp_means[batch_indices, indices, :].squeeze(1)

            loss = F.mse_loss(pred_outputs, outputs, reduction='mean')
            test_losses.append(loss.item())
            gatings.append(gating)

        logging_dict['test_mean_mse'] = sum(test_losses)/len(test_losses)

        gatings = ch.cat(gatings, dim=0)
        avrg_entropy = ch.distributions.Categorical(probs=gatings).entropy().mean().item()
        logging_dict['test_mean_gating_entropy'] = avrg_entropy
        # for i in range(self.agent.n_components):
        #     average_gating_prob = gatings[:, i].mean().item()
        #     logging_dict[f'cmp_{i}_aver_gating_prob'] = average_gating_prob

        return logging_dict


    def armotized_vi_update(self, inputs, actions, goals=None):

        pred_means, pred_chols, pred_gatings = self.agent.forward(inputs, goals)
        pred_log_gatings = pred_gatings.log()

        pred_means = einops.rearrange(pred_means, 'b c t a -> (b t) c a')
        pred_chols = einops.rearrange(pred_chols, 'b c t a1 a2 -> (b t) c a1 a2')
        pred_log_gatings = einops.rearrange(pred_log_gatings, 'b c t -> (b t) c')
        actions = einops.rearrange(actions, 'b t a -> (b t) a')

        with ch.no_grad():
            actions = actions[:, None, :].repeat(1, self.agent.n_components, 1)
            log_probs = self.agent.joint_cmps.log_probability((pred_means, pred_chols), actions)
            ### log the log_probs per component
            log_resps = log_probs + pred_log_gatings
            log_resps = log_resps - ch.logsumexp(log_resps, dim=1, keepdim=True)

        loss_fn = SoftCrossEntropyLoss()

        targets = log_resps.exp() + 1e-8

        if torch.isnan(log_probs).any():
            print("Nan in log probs")
            print(log_probs)
        if torch.isnan(log_resps).any():
            print("Nan in log resps")
            print(log_resps)
        if torch.isnan(pred_log_gatings).any():
            print("Nan in pred log gatings")
            print(pred_log_gatings)
        if torch.isnan(targets).any():
            print("Nan in gating targets")
            print(targets)

        loss = loss_fn(pred_log_gatings, targets)
        self.gating_optimizer.zero_grad()
        loss.backward()
        self.gating_optimizer.step()
        if self.gating_lr_scheduler is not None:
            self.gating_lr_scheduler.step()

        ret_dict = {'gating_loss': loss.item()}

        return ret_dict


    def iterative_train_cmp(self, batch, iter):
        """
        Train the joint GMM policy
        b -- state batch
        c -- the number of components
        v -- the number of vi samples
        a -- action dimension
        o -- observation dimension
        """
        if not self.is_vision_task:
            states = batch[0].to(self.device).to(self.dtype)
        # states = self.scaler.scale_input(states)
        else:
            states = batch[0]

        if batch[2] is not None:
            goals = batch[2].to(self.device).to(self.dtype)
        else:
            goals = None


        logging_dict = {}

        # input : (b, t, o)
        # pred_means, pred_chols : (b, c, t, a), (b, c, t, a, a)
        pred_means, pred_chols, pred_gatings = self.agent(states, goals)

        ###FIXME: hack for not training chols
        if self.detach_chol:
            pred_chols = pred_chols.detach()

        pred_gatings = pred_gatings.detach()

        # sampled actions : (v, b, c, t, a)
        sampled_actions = self.agent.joint_cmps.rsample((pred_means, pred_chols), n=self.vi_batch_size)

        # rearrange the sampled actions to (b, c, v, t, a)
        if len(sampled_actions.size()) == 4:
            # sampled_actions = sampled_actions.permute(1, 2, 0, 3)
            sampled_actions = einops.rearrange(sampled_actions, 'v b c a -> b c v a')

        elif len(sampled_actions.size()) == 5:
            sampled_actions = einops.rearrange(sampled_actions, 'v b c t a -> b c v t a')


        # Query the scores function
        # input : states (b, c, v, t, o), actions (b, c, v, t, a)
        # output : scores (b, c, v, a)
        if self.is_vision_task:
            # for vision task first encode the states and then repeat the latent states to save VRAM
            score_states = states
        else:
            score_states = einops.repeat(states, 'b t o -> b c v t o', c=self.agent.n_components, v=self.vi_batch_size)

        score_goals = einops.repeat(goals, 'b t o -> b c v t o', c=self.agent.n_components, v=self.vi_batch_size) if goals is not None else None
        with ch.no_grad():
            scores, noise_level = self.score_function(sampled_actions, score_states, score_goals, iter, self.is_vision_task)

        ### pack the scores to ((b,t), c, v, a)
        scores = einops.rearrange(scores, 'b c v t a -> (b t) c v a')
        sampled_actions = einops.rearrange(sampled_actions, 'b c v t a -> (b t) c v a')
        pred_means = einops.rearrange(pred_means, 'b c t a -> (b t) c a')
        pred_chols = einops.rearrange(pred_chols, 'b c t a1 a2 -> (b t) c a1 a2')
        pred_gatings = einops.rearrange(pred_gatings, 'b c t -> (b t) c')

        ### TODO: check if the multiplication is correct( should that be element-wise multiplication?)
        # score dot action : (b, c, v)
        # score_w_act = torch.einsum('bcva,bcva->bcv', scores, sampled_actions)
        score_w_act = torch.einsum('...va,...va->...v', scores, sampled_actions)

        # log responsibilities : (b, c, v)
        responsibilities = self.agent.log_responsibilities(pred_means.clone().detach(),
                                                           pred_chols.clone().detach(),
                                                           pred_gatings,
                                                           sampled_actions)

        # entropies : (b, c)
        entropies = self.agent.joint_cmps.entropy((pred_means, pred_chols))

        # expectation for r_sample_terms: (b, c)
        r_sample_term = (score_w_act + responsibilities).mean(dim=-1)

        ### FIXME: weighted update
        unweighted_vi_loss = r_sample_term + entropies
        vi_loss = - (unweighted_vi_loss * pred_gatings).mean()

        self.cmps_optimizer.zero_grad(set_to_none=True)
        vi_loss.backward()
        self.cmps_optimizer.step()
        if self.cmps_lr_scheduler is not None:
            self.cmps_lr_scheduler.step()

        logging_dict[f"vi_loss"] = vi_loss.item()
        logging_dict[f"score_w_loss"] = -score_w_act.mean().item()
        logging_dict[f"responsibility_loss"] = -responsibilities.mean().item()
        logging_dict[f"entropy_loss_cmp"] = -entropies.mean().item()
        logging_dict[f"noise_level"] = noise_level.float().mean().item()

        if torch.isnan(vi_loss).any():
            print("vi loss is nan")
            if torch.isnan(score_w_act).any():
                print("score_w_act is nan")
            if torch.isnan(responsibilities).any():
                print("responsibilities is nan")
            if torch.isnan(entropies).any():
                print("entropies is nan")
            if torch.isnan(pred_gatings).any():
                print("pred_gatings is nan")
            exit(0)

        return logging_dict

    @staticmethod
    def create_vid_agent(policy_params, optimizer_params, training_params, score_function,
                         train_dataset=None, test_dataset=None):

        policy = JointGaussianMixtureModel(**policy_params, device=training_params['device'], dtype=training_params['dtype'])

        if policy_params['vision_task']:
            if policy_params['copy_vision_encoder']:
                policy.vision_encoder = copy.deepcopy(score_function.model.model.obs_encoder)
            if policy_params['train_vision_encoder']:
                cmps_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                              model_parameters=list(policy.joint_cmps.parameters())+list(policy.vision_encoder.parameters()),
                                              learning_rate=optimizer_params['cmps_lr'],
                                              weight_decay=optimizer_params['cmps_weight_decay'])
            else:
                cmps_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                               model_parameters=policy.joint_cmps.parameters(),
                                               learning_rate=optimizer_params['cmps_lr'],
                                               weight_decay=optimizer_params['cmps_weight_decay'])
        else:
            cmps_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                           model_parameters=policy.joint_cmps.parameters(),
                                           learning_rate=optimizer_params['cmps_lr'],
                                           weight_decay=optimizer_params['cmps_weight_decay'])

        cmps_lr_scheduler = get_lr_schedule(optimizer_params['cmps_lr_schedule'],
                                           cmps_optimizer, training_params['max_train_iters']) \
            if optimizer_params['cmps_lr_schedule'] is not None else None

        if policy_params['learn_gating']:
            gating_net_optimizer = get_optimizer(optimizer_type=optimizer_params['optimizer_type'],
                                                 model_parameters=policy.gating_network.parameters(),
                                                 learning_rate=optimizer_params['gating_lr'],
                                                 weight_decay=optimizer_params['gating_weight_decay'])
            gating_lr_scheduler = get_lr_schedule(optimizer_params['gating_lr_schedule'],
                                                    gating_net_optimizer, training_params['max_train_iters']) \
                    if optimizer_params['gating_lr_schedule'] is not None else None
        else:
            gating_net_optimizer = None
            gating_lr_scheduler = None

        vid_agent = AmortizedVID_jointGMM(agent=policy, cmps_optimizer=cmps_optimizer, gating_optimizer=gating_net_optimizer,
                                          cmps_lr_scheduler=cmps_lr_scheduler,
                                          gating_lr_scheduler=gating_lr_scheduler,
                                          score_function=score_function,
                                          train_dataloader=train_dataset,
                                          test_dataloader=test_dataset,
                                          learn_gating=policy_params['learn_gating'],
                                          detach_chol=policy_params['detach_chol'],
                                          **training_params)
        return vid_agent

    def save_best_model(self, path):
        """
        save the model
        :return:
        """
        ch.save(self.agent, os.path.join(path, f"best_model.pt"))

    def save_model(self, iteration, path):
        save_path = os.path.join(path, f"model_state_dict_{iteration}.pth")
        ch.save(self.agent.state_dict(), save_path)

    def save_debug_model(self, path):
        """
        save the model
        :return:
        """
        ch.save(self.agent, os.path.join(path, f"debug_model.pt"))

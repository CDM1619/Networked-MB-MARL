# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 01:52:16 2022

@author: 86153
"""


import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
# from algorithms.algorithm import ReplayBuffer
from ray.state import actors
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import pickle
from copy import deepcopy as dp
from algorithms.models import CategoricalActor, EnsembledModel, SquashedGaussianActor, ParameterizedModel_MBPPO
import random
import multiprocessing as mp
from torch import distributed as dist
import argparse
from algorithms.algo.buffer import MultiCollect,Trajectory,TrajectoryBuffer,ModelBuffer


class IC3Net(nn.ModuleList):
    def __init__(self, logger, device, agent_args, input_args, **kwargs):
        super().__init__()
        self.logger = logger
        self.device = device
        self.n_agent = agent_args.n_agent
        self.gamma = agent_args.gamma
        self.lamda = agent_args.lamda
        self.clip = agent_args.clip
        self.target_kl = agent_args.target_kl
        self.v_coeff = agent_args.v_coeff
        self.v_thres = agent_args.v_thres
        self.entropy_coeff = agent_args.entropy_coeff
        self.entropy_coeff_decay =  agent_args.entropy_coeff_decay  # add
        self.lr = agent_args.lr
        self.lr_v = agent_args.lr_v
        self.lr_p = agent_args.lr_p
        self.n_update_v = agent_args.n_update_v
        self.n_update_pi = agent_args.n_update_pi
        self.n_minibatch = agent_args.n_minibatch
        self.use_reduced_v = agent_args.use_reduced_v
        self.use_rtg = agent_args.use_rtg
        self.use_gae_returns = agent_args.use_gae_returns
        self.all_comm = True # TODO: need to update
        self.advantage_norm = agent_args.advantage_norm
        self.observation_dim = agent_args.observation_dim
        self.action_space = agent_args.action_space
        self.discrete = isinstance(agent_args.action_space, Discrete)
        self.env_name = input_args.env
        if self.discrete:
            self.action_dim = self.action_space.n
            self.action_shape = self.action_dim
        else:
            self.action_shape = self.action_space.shape
            self.action_dim = self.action_space.shape[0]
            self.action_low = self.action_space.low
            self.action_high = self.action_space.high
            self.squeeze = agent_args.squeeze
        # if adj diag is not one, we should add a eye matrix
        agent_args.adj = (torch.as_tensor(agent_args.adj, device=self.device, dtype=torch.float) > 0) | torch.eye(
            self.n_agent, device=device).bool()
        self.adj = torch.as_tensor(agent_args.adj, device=self.device, dtype=torch.float)
        self.radius_v = agent_args.radius_v
        self.radius_pi = agent_args.radius_pi
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args
        #self.collect_pi, self.actors = self._init_actors()
        #self.collect_v, self.vs = self._init_vs()
        self.hidden_dim = agent_args.v_args.hidden_dim
        self.initNetwork(agent_args)
        
        #self.actors.load_state_dict(torch.load(test_actors_model, map_location={'cuda:0':'cuda:0','cuda:1':'cuda:0','cuda:2':'cuda:0','cuda:3':'cuda:0','cuda:4':'cuda:0','cuda:5':'cuda:0','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu','cuda:4':'cpu','cuda:5':'cpu',}))  
        
        self.optimizer = Adam(list(self.obs_encoder.parameters())+list(self.comm_gate_head.parameters())+ list(self.message_models.parameters())+list(self.main_models.parameters())+list(self.value_heads.parameters())+list(self.actors.parameters()), lr=self.lr)
        #self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)

    def initNetwork(self, agent_args):

        # [one for all]observation encoding layer
        self.obs_encoder = nn.Linear(self.observation_dim, self.hidden_dim).to(self.device)

        # [1 v 1] communication gated action layer, first dim for comm, second dim for not
        self.comm_gate_head = nn.ModuleList([nn.Linear(self.hidden_dim, 2).to(self.device) for i in range(self.n_agent)])

        # [1 v 1] message fussion layer
        self.message_models = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device) for i in range(self.n_agent)])

        # [1 v 1] main layer
        self.main_models = nn.ModuleList([nn.Linear(self.hidden_dim, self.hidden_dim).to(self.device) for i in range(self.n_agent)])

        # value head
        self.value_heads = nn.ModuleList([nn.Linear(self.hidden_dim, 1).to(self.device) for i in range(self.n_agent)])



        # action head
        if self.discrete:
            self.pi_args.sizes[0] = self.hidden_dim
            self.actors = nn.ModuleList([CategoricalActor(**self.pi_args._toDict()).to(self.device) for i in range(self.n_agent)])
        else:
            self.pi_args.sizes[0] = self.hidden_dim
            self.actors = nn.ModuleList([GaussianActor(action_dim=self.action_dim, **self.pi_args._toDict()).to(self.device) for i in range(self.n_agent)])

        # activation function
        self.activation_function = torch.nn.ReLU(inplace=True) #TODO: add to config file


    def updateAgent(self, trajs, clip=None):
        time_t = time.time()
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = Trajectory.names()
        traj_all = {name: [] for name in names}
        max_traj_length = max([i.length for i in trajs])
        for traj in trajs:
            for name in names:
                tensor_shape = traj[name].shape
                full_part_shape = [max_traj_length - tensor_shape[0]] + list(tensor_shape[1:])
                if name == 'd':
                    traj_all[name].append(torch.cat([traj[name], torch.ones(full_part_shape, dtype=torch.bool, device=self.device)], dim=0))
                else:
                    traj_all[name].append(
                        torch.cat([traj[name], torch.zeros(full_part_shape, dtype=traj[name].dtype, device=self.device)], dim=0))
        # should be 4-dim [batch * step * n_agent * dim]
        traj = {name: torch.stack(value, dim=0) for name, value in traj_all.items()}

        s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
        s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
        # all in shape [batch_size, T, n_agent, dim]
        value_old, returns, advantages, reduced_advantages = self._process_traj(**traj)

        advantages_old = reduced_advantages if self.use_reduced_v else advantages  # set use_reduced_v as False

        b, T, n, d_s = s.size()
        d_a = a.size()[-1]
        s = s.view(-1, n, d_s)
        a = a.view(-1, n, d_a)
        logp = logp.view(-1, n, d_a)
        advantages_old = advantages_old.view(-1, n, 1)
        returns = returns.view(-1, n, 1)
        value_old = value_old.view(-1, n, 1)
        # s, a, logp, adv, ret, v are now all in shape [-1, n_agent, dim]

        batch_total = logp.size()[0]
        batch_size = int(batch_total / n_minibatch)

        # critic loss
        batch_returns = returns
        batch_state = s
        if n_minibatch > 1:
            idxs = np.random.randint(0, len(batch_total), size=batch_size)
            [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
        batch_v_new = self._evalV(batch_state)
        loss_v = ((batch_v_new - batch_returns) ** 2).mean()
        var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
        rel_v_loss = loss_v / (var_v + 1e-8)
        self.logger.log(v_loss=loss_v, v_update=None, v_var=var_v, rel_v_loss=rel_v_loss)
        # if rel_v_loss < self.v_thres:
        #     break
        self.logger.log(v_update_step=1)
        
        
        #--------------------------------
        #self.logger.log(reward=r)
        self.logger.log(reward=r,returns=returns)
        #--------------------------------
        
        
        # actor loss
        batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
        if n_minibatch > 1:
            idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
            [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in
                                                                             [batch_state, batch_action, batch_logp,
                                                                              batch_advantages_old]]
        batch_logp_new = self.get_logp(batch_state, batch_action)

        # - A * logp - entropy_loss
        loss_pi = torch.mean(- batch_advantages_old * batch_logp_new)
        loss_entropy = - torch.mean(batch_logp_new)
        updata_entropy_coff = max(self.entropy_coeff - self.entropy_coeff_decay * self.logger.buffer['interaction'],
                                  0)
        loss_actor = loss_pi + loss_entropy * updata_entropy_coff

        loss = self.lr_v * loss_v + self.lr_p * loss_actor

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        logp_diff = torch.exp(batch_logp_new) * (batch_logp_new - self.get_logp(batch_state, batch_action))
        kl = logp_diff.mean()
        self.logger.log(pi_loss=loss_pi, entropy=loss_entropy, kl_divergence=kl, entropy_coff=updata_entropy_coff,
                        pi_update=None)
        self.logger.log(pi_update_step=1)

    def checkConverged(self, ls_info):
        #TODO: not neccessary
        return False

    def group_inference(self, model, data):
        # model is a modelList
        # data is a [batch * n_agent *dim]
        outs = []
        for i in range(self.n_agent):
            agent_data = data.select(1, i)
            outs.append(model[i](agent_data))
        outs = torch.stack(outs, dim=1)
        return outs

    def inference_hidden_state(self, s):
        # encode the state
        s=s.to(self.device)
        s_encoding = self.activation_function(self.obs_encoder(s))

        # decide which agent to communication
        # [batch_size, n_agent, 2]
        comm_gate_distirbution = self.group_inference(self.comm_gate_head, s_encoding)


        # merge the message by mean
        if self.all_comm == True:
            batch_comm_adj = self.adj.unsqueeze(0).repeat(s.shape[0], 1, 1)
        else:
            # TODO: check if need detach
            comm_gate = torch.stack(
                [torch.stack([torch.multinomial(dis, 1).detach() for dis in n_dis]) for n_dis in
                 comm_gate_distirbution])
            comm_gate_tensor = comm_gate.unsqueeze(2).repeat(1, 1, self.n_agent)
            comm_gate_tensor_T = comm_gate_tensor.permute(0, 2, 1)
            comm_gate = comm_gate_tensor * comm_gate_tensor_T
            batch_comm_adj = self.adj.unsqueeze(0).repeat(comm_gate.shape[0], 1, 1) | comm_gate

        batch_message_ls = []
        for i in range(s.shape[0]):
            # for each batch
            comm_adj = batch_comm_adj.select(0, i)
            single_s_encoding = s_encoding.select(0, i)

            batch_message_ls.append(torch.mm(comm_adj, single_s_encoding))
        batch_message = torch.stack(batch_message_ls, dim=0)

        # deal with the meassage
        deal_message = self.group_inference(self.message_models, batch_message)

        # genarate hidden state
        hidden_state = s_encoding + self.group_inference(self.main_models, s_encoding) + deal_message
        return hidden_state

    def save(self, info=None):
        self.logger.save(self, info=info)

    def save_nets(self, dir_name,episode):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        # torch.save(self.critic.state_dict(), dir_name + '/Models/' +str(episode)+ 'best_critic.pt')
        torch.save(self.actors.state_dict(), dir_name + '/Models/' + str(episode)+ 'best_actor.pt')
        # torch.save(self.actors.state_dict(), dir_name + '/' +str(episode)+ 'best_actor.pt')
        print('RL saved successfully')

    def act(self, s, if_test=False, requires_log=False):
        """
                Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
                This method is gradient-free. To get the gradient-enabled probability information, use get_logp().
                Returns a distribution with the same dimensions of input.
                """
        with torch.no_grad():
            dim = s.dim()
            while s.dim() <= 2:
                s = s.unsqueeze(0)
            s = s.to(self.device)

            hidden_state = self.inference_hidden_state(s)
            hidden_state =  hidden_state.permute(1, 0, 2) # Now s[i].dim() == 2 ([batch_size, dim])

            # cal the action
            if self.discrete:
                probs = []
                for i in range(self.n_agent):
                    probs.append(self.actors[i](hidden_state[i]))
                probs = torch.stack(probs, dim=1)
                return Categorical(probs)


            else:
                means, stds = [], []
                for i in range(self.n_agent):
                    mean, std = self.actors[i](hidden_state[i])
                    means.append(mean)
                    stds.append(std)
                means = torch.stack(means, dim=1)
                stds = torch.stack(stds, dim=1)
                while means.dim() > dim:
                    means = means.squeeze(0)
                    stds = stds.squeeze(0)
                return Normal(means, stds)

    def get_logp(self, s, a):
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        dim = s.dim()
        while s.dim() <= 2:
            s = s.unsqueeze(0)
            a = a.unsqueeze(0)
        while a.dim() < s.dim():
            a = a.unsqueeze(-1)

        hidden_state = self.inference_hidden_state(s)
        hidden_state = hidden_state.permute(1, 0, 2)  # Now s[i].dim() == 2 ([batch_size, dim])

        log_prob = []
        for i in range(self.n_agent):
            if self.discrete:
                probs = self.actors[i](hidden_state[i])
                log_prob.append(torch.log(torch.gather(probs, dim=-1, index=torch.select(a, dim=1, index=i).long())))
            else:
                log_prob.append(self.actors[i](hidden_state[i], a.select(dim=1, index=i)))
        log_prob = torch.stack(log_prob, dim=1)
        while log_prob.dim() < 3:
            log_prob = log_prob.unsqueeze(-1)
        return log_prob

    def _evalV(self, s):
        # TODO：
        # Requires input in shape [-1, n_agent, dim]
        s = s.to(self.device)

        hidden_state = self.inference_hidden_state(s)
        hidden_state = hidden_state.permute(1, 0, 2)  # Now s[i].dim() == 2 ([batch_size, dim])

        values = []
        for i in range(self.n_agent):
            values.append(self.value_heads[i](hidden_state[i]))
        return torch.stack(values, dim=1)

    def _process_traj(self, s, a, r, s1, d, logp):
        """
        Input are all in shape [batch_size, T, n_agent, dim]
        """
        with torch.no_grad():
            b, T, n, dim_s = s.shape
            s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
            value = self._evalV(s.view(-1, n, dim_s)).view(b, T, n, -1)

            returns = torch.zeros(value.size(), device=self.device)
            deltas, advantages = torch.zeros_like(returns), torch.zeros_like(returns)
            prev_value = self._evalV(s1.select(1, T - 1))
            if not self.use_rtg:
                prev_return = prev_value
            else:
                prev_return = torch.zeros_like(prev_value)
            prev_advantage = torch.zeros_like(prev_return)
            d_mask = d.float()
            for t in reversed(range(T)):
                deltas[:, t, :, :]= r.select(1, t) + self.gamma * (1-d_mask.select(1,t)) * prev_value - value.select(1, t).detach()
                advantages[:, t, :, :] = deltas.select(1, t) + self.gamma * self.lamda * (1-d_mask.select(1,t)) * prev_advantage
                if self.use_gae_returns:
                    returns[:, t, :, :] = value.select(1, t).detach() + advantages.select(1, t)
                else:
                    returns[:, t, :, :] = r.select(1, t) + self.gamma * (1-d_mask.select(1, t)) * prev_return

                prev_return = returns.select(1, t)
                prev_value = value.select(1, t)
                prev_advantage = advantages.select(1, t)

        return value, returns, advantages, None

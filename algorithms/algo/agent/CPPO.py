# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:04:27 2022

@author: 86153
"""

import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, GraphConvolutionalModel, MLP, CategoricalActor
from tqdm.std import trange
#from algorithms.algorithm import ReplayBuffer
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


class CPPOAgent(nn.ModuleList):
    """
    Everything in and out is torch Tensor.
    """
    def __init__(self, logger, device, agent_args, env_args, **kwargs):
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
        self.lr = agent_args.lr
        self.lr_v = agent_args.lr_v
        self.n_update_v = agent_args.n_update_v
        self.n_update_pi = agent_args.n_update_pi
        self.n_minibatch = agent_args.n_minibatch
        self.use_reduced_v = agent_args.use_reduced_v
        self.use_rtg = agent_args.use_rtg
        self.use_gae_returns = agent_args.use_gae_returns
        self.env_name = env_args.env
        self.algo_name = env_args.algo
        self.advantage_norm = agent_args.advantage_norm
        self.observation_dim = agent_args.observation_dim
        self.action_space = agent_args.action_space
        self.discrete = isinstance(agent_args.action_space, Discrete)       
        if self.discrete:
            self.action_dim = self.action_space.n
            self.action_shape = self.action_dim
            
        else:
            self.action_shape = self.action_space.shape             
            self.action_dim = self.action_space.shape[0]
            self.squeeze = agent_args.squeeze
                 
        self.adj = torch.as_tensor(agent_args.adj, device=self.device, dtype=torch.float)
        self.radius_v = agent_args.radius_v
        self.radius_pi = agent_args.radius_pi
        self.pi_args = agent_args.pi_args
        self.v_args = agent_args.v_args
        self.collect_pi, self.actors = self._init_actors()
        self.collect_v, self.vs = self._init_vs()

        #self.actors.load_state_dict(torch.load(test_actors_model, map_location={'cuda:0':'cuda:0','cuda:1':'cuda:0','cuda:2':'cuda:0','cuda:3':'cuda:0','cuda:4':'cuda:0','cuda:5':'cuda:0','cuda:0':'cpu','cuda:1':'cpu','cuda:2':'cpu','cuda:3':'cpu','cuda:4':'cpu','cuda:5':'cpu',})) 

        self.optimizer_v = Adam(self.vs.parameters(), lr=self.lr_v)
        self.optimizer_pi = Adam(self.actors.parameters(), lr=self.lr)

    def act(self, s,if_test=False, requires_log=False):
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
            s = self.collect_pi.gather(s) # all state into [ self +  ]
            # Now s[i].dim() == 2 ([batch_size, dim])

            if self.discrete:
                if if_test: 
                    probs = []
                    for i in range(self.n_agent):
                        index = torch.argmax(self.actors[i](s[i])[0])
                        action_p = torch.tensor([[0]*self.action_dim])
                        action_p[0][index] = 1
                        probs.append(action_p)
                    probs = torch.stack(probs, dim=1)
                    while probs.dim() > dim:
                        probs = probs.squeeze(0)
                    return Categorical(probs)
                  
                else:
                    probs = []
                    for i in range(self.n_agent):
                        probs.append(self.actors[i](s[i]))
                    probs = torch.stack(probs, dim=1)
                    while probs.dim() > dim:
                        probs = probs.squeeze(0)
                    return Categorical(probs)

            else:
                means, stds = [], []
                for i in range(self.n_agent):
                    mean, std = self.actors[i](s[i])
                    means.append(mean)
                    

                    stds.append(std)
                    # stds.append(std.exp())
                    
                    
                    
                means = torch.stack(means, dim=1)
                stds = torch.stack(stds, dim=1)
                while means.dim() > dim:
                    means = means.squeeze(0)
                    stds = stds.squeeze(0)
                return Normal(means, stds)
    
    def get_logp(self, s, a):
        """
        Requires input of [batch_size, n_agent, dim] or [n_agent, dim].
        Returns a tensor whose dim() == 3.
        """
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        dim = s.dim()

        while s.dim() <= 2:
            s = s.unsqueeze(0)
            a = a.unsqueeze(0)
        while a.dim() < s.dim():
            a = a.unsqueeze(-1)
            
        s = self.collect_pi.gather(s)
        # Now s[i].dim() == 2, a.dim() == 3
        log_prob = []
        for i in range(self.n_agent):
            if self.discrete:
                probs = self.actors[i](s[i])
                log_prob.append(torch.log(torch.gather(probs, dim=-1, index=torch.select(a, dim=1, index=i).long())))
            else:
                log_prob.append(self.actors[i](s[i], a.select(dim=1, index=i)))
        log_prob = torch.stack(log_prob, dim=1)
        while log_prob.dim() < 3:
            log_prob = log_prob.unsqueeze(-1)
        return log_prob

    def updateAgent(self, trajs, clip=None):
        time_t = time.time()
        if clip is None:
            clip = self.clip
        n_minibatch = self.n_minibatch

        names = Trajectory.names()
        traj_all = {name:[] for name in names}
        max_traj_length = max([i.length for i in trajs])
        for traj in trajs:
            for name in names:
                tensor_shape = traj[name].shape
                full_part_shape = [max_traj_length - tensor_shape[0]] + list(tensor_shape[1:])
                if name == 'd':
                    traj_all[name].append(torch.cat([traj[name], torch.ones(full_part_shape, dtype=torch.bool, device=self.device)], dim=0))
                else:
                    traj_all[name].append(torch.cat([traj[name], torch.zeros(full_part_shape, dtype=traj[name].dtype, device=self.device)], dim=0))
        traj = {name:torch.stack(value, dim=0) for name, value in traj_all.items()}

        for i_update in range(self.n_update_pi):
            s, a, r, s1, d, logp = traj['s'], traj['a'], traj['r'], traj['s1'], traj['d'], traj['logp']
            s, a, r, s1, d, logp = [item.to(self.device) for item in [s, a, r, s1, d, logp]]
            # all in shape [batch_size, T, n_agent, dim]
            value_old, returns, advantages, reduced_advantages = self._process_traj(**traj)
            

            
            advantages_old = reduced_advantages if self.use_reduced_v else advantages

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
            batch_size = int(batch_total/n_minibatch)


            kl_all = []
            i_pi = 0
            for i_pi in range(1):  
                batch_state, batch_action, batch_logp, batch_advantages_old = [s, a, logp, advantages_old]
                if n_minibatch > 1:
                    idxs = np.random.choice(range(batch_total), size=batch_size, replace=False)
                    [batch_state, batch_action, batch_logp, batch_advantages_old] = [item[idxs] for item in [batch_state, batch_action, batch_logp, batch_advantages_old]]
                batch_logp_new = self.get_logp(batch_state, batch_action)
            
                
                logp_diff = batch_logp_new - batch_logp
                kl = logp_diff.mean()
                ratio = torch.exp(batch_logp_new - batch_logp)
                surr1 = ratio * batch_advantages_old
                surr2 = ratio.clamp(1 - clip, 1 + clip) * batch_advantages_old
                loss_surr = torch.min(surr1, surr2).mean()
                loss_entropy = - torch.mean(batch_logp_new)
                loss_pi = - loss_surr - self.entropy_coeff * loss_entropy
                self.optimizer_pi.zero_grad()
                loss_pi.backward()
                self.optimizer_pi.step()
                self.logger.log(surr_loss = loss_surr, entropy = loss_entropy, kl_divergence = kl, pi_update=None)
                kl_all.append(kl.abs().item())
                if self.target_kl is not None and kl.abs() > 1.5 * self.target_kl:
                    break
            self.logger.log(pi_update_step=i_update)

            for i_v in range(1):
                batch_returns = returns
                batch_state = s
                if n_minibatch > 1:
                    idxs = np.random.randint(0, len(batch_total), size=batch_size)
                    [batch_returns, batch_state] = [item[idxs] for item in [batch_returns, batch_state]]
                batch_v_new = self._evalV(batch_state)
                loss_v = ((batch_v_new - batch_returns) ** 2).mean()
                self.optimizer_v.zero_grad()
                loss_v.backward()
                self.optimizer_v.step()
            
                var_v = ((batch_returns - batch_returns.mean()) ** 2).mean()
                rel_v_loss = loss_v / (var_v + 1e-8)
                self.logger.log(v_loss=loss_v, v_update=None, v_var=var_v, rel_v_loss=rel_v_loss)
                if rel_v_loss < self.v_thres:
                    break
            self.logger.log(v_update_step=i_update)
            self.logger.log(update=None, reward=r, value=value_old, clip=clip, returns=returns, advantages=advantages_old.abs())
        self.logger.log(agent_update_time=time.time()-time_t)
        return [r.mean().item(), loss_entropy.item(), max(kl_all)]
    
    def checkConverged(self, ls_info):
        return False

    def save(self, info=None):
        self.logger.save(self, info=info)

    def load(self, state_dict):
        self.load_state_dict(state_dict[self.logger.prefix])
        
#_____________________________________________________________________________________________________--

    def save_nets(self, dir_name,episode):
        if not os.path.exists(dir_name + '/Models'):
            os.mkdir(dir_name + '/Models')
        # torch.save(self.critic.state_dict(), dir_name + '/Models/' +str(episode)+ 'best_critic.pt')
        torch.save(self.actors.state_dict(), dir_name + '/Models/' + str(episode)+ 'best_actor.pt')
        # torch.save(self.actors.state_dict(), dir_name + '/' +str(episode)+ 'best_actor.pt')
        print('RL saved successfully')

    def load_nets(self, dir_name,episode):       
        self.actors.load_state_dict(torch.load(dir_name + '/Models/' + str(episode)+ 'best_actor.pt'))
#_____________________________________________________________________________________________________--
        

    def _evalV(self, s):
        # Requires input in shape [-1, n_agent, dim]
        s = s.to(self.device)
        s = self.collect_v.gather(s)
        values = []
        for i in range(self.n_agent):
            values.append(self.vs[i](s[i]))
        return torch.stack(values, dim=1)

    def _init_actors(self):
        if self.env_name == 'UAV_9d' and self.algo_name == 'CPPO':
            self.adj = torch.as_tensor(np.ones((25, 25)), device=self.device, dtype=torch.float)
        
        collect_pi = MultiCollect(torch.matrix_power(self.adj, self.radius_pi), device=self.device)
        actors = nn.ModuleList()
        for i in range(self.n_agent):
            self.pi_args.sizes[0] = collect_pi.degree[i] * self.observation_dim
            if self.discrete:
                actors.append(CategoricalActor(**self.pi_args._toDict()).to(self.device))
            else:
                actors.append(GaussianActor(action_dim=self.action_dim, **self.pi_args._toDict()).to(self.device))
        return collect_pi, actors
    
    def _init_vs(self):
        if self.env_name == 'UAV_9d':
            self.adj = torch.as_tensor(np.ones((25, 25)), device=self.device, dtype=torch.float)
            
        collect_v = MultiCollect(torch.matrix_power(self.adj, self.radius_v), device=self.device)
        vs = nn.ModuleList()
        for i in range(self.n_agent):
            self.v_args.sizes[0] = collect_v.degree[i] * self.observation_dim
            #print('self.v_args.sizes[0]=',self.v_args.sizes[0])
            v_fn = self.v_args.network
            vs.append(v_fn(**self.v_args._toDict()).to(self.device))
        return collect_v, vs
    
    def _process_traj(self, s, a, r, s1, d, logp):
        """
        Input are all in shape [batch_size, T, n_agent, dim]
        """
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
        reduced_advantages = self.collect_v.reduce_sum(advantages.view(-1, n, 1)).view(advantages.size())
        if self.advantage_norm and reduced_advantages.size()[1] > 1:
            reduced_advantages = (reduced_advantages - reduced_advantages.mean(dim=1, keepdim=True)) / (reduced_advantages.std(dim=1, keepdim=True) + 1e-5)
            advantages = (advantages - advantages.mean(dim=1, keepdim=True)) / (advantages.std(dim=1, keepdim=True) + 1e-5)
        return value.detach(), returns, advantages.detach(), reduced_advantages.detach()

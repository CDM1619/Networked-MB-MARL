# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 02:09:45 2022

@author: 86153
"""


import time
import os
from numpy.core.numeric import indices
from torch.distributions.normal import Normal
from algorithms.utils import collect, mem_report
from algorithms.models import GaussianActor, MLP, CategoricalActor
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
from algorithms.algo.agent.DPPO import DPPOAgent
from algorithms.algo.buffer import MultiCollect,Trajectory,TrajectoryBuffer,ModelBuffer
from algorithms.models import GraphConvolutionalModel

class ModelBasedAgent(nn.ModuleList):
    def __init__(self, logger, device, agent_args,env_args, **kwargs):
        super().__init__(logger, device, agent_args,env_args, **kwargs)
        self.logger = logger
        self.device = device
        self.lr_p = agent_args.lr_p
        self.p_args = agent_args.p_args
        self.ps = GraphConvolutionalModel(self.logger, self.adj, self.observation_dim, self.action_dim, self.n_agent, self.p_args).to(self.device)
        self.optimizer_p = Adam(self.ps.parameters(), lr=self.lr)


    def updateModel(self, trajs, length=1):
        """
        Input dim: 
        s: [[T, n_agent, state_dim]]
        a: [[T, n_agent, action_dim]]
        """
             
        time_t = time.time()
        loss_total = 0.
        ss, actions, rs, s1s, ds = [], [], [], [], []
        for traj in trajs:
            s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
            s, a, r, s1, d = [torch.as_tensor(item, device=self.device) for item in [s, a, r, s1, d]]          
            ss.append(s)
            actions.append(a)
            rs.append(r)
            s1s.append(s1)
            ds.append(d)  
            
        ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
        loss, rel_state_error = self.ps.train(ss, actions, rs, s1s, ds, length) # [n_traj, T, n_agent, dim]
     
        self.optimizer_p.zero_grad()
        loss.sum().backward()
        # torch.nn.utils.clip_grad_norm_(parameters=self.ps.parameters(), max_norm=5, norm_type=2)
        self.optimizer_p.step()
#——————————————————————————————————————————————————————————————————————————————————        
        self.logger.log(p_loss_total=loss.sum(), p_update=None)
        self.logger.log(model_update_time=time.time()-time_t)
#——————————————————————————————————————————————————————————————————————————————————   
        return rel_state_error.item()
    
    def validateModel(self, trajs, length=1):
        with torch.no_grad():
            ss, actions, rs, s1s, ds = [], [], [], [], []
            for traj in trajs:
                s, a, r, s1, d = traj["s"], traj["a"], traj["r"], traj["s1"], traj["d"]
                s, a, r, s1, d = [torch.as_tensor(item, device=self.device) for item in [s, a, r, s1, d]]
                ss.append(s)
                actions.append(a)
                rs.append(r)
                s1s.append(s1)
                ds.append(d)
            ss, actions, rs, s1s, ds = [torch.stack(item, dim=0) for item in [ss, actions, rs, s1s, ds]]
            _, rel_state_error = self.ps.train(ss, actions, rs, s1s, ds, length) # [n_traj, T, n_agent, dim]
            return rel_state_error.item()
    
    def model_step(self, s, a):
        """
        Input dim: 
        s: [batch_size, n_agent, state_dim]
        a: [batch_size, n_agent] (discrete) or [batch_size, n_agent, action_dim] (continuous)

        Return dim == 3.
        """
        with torch.no_grad():
            while s.dim() <= 2:
                s = s.unsqueeze(0)
                a = a.unsqueeze(0)
            while a.dim() <= 2:
                a = a.unsqueeze(-1)
            s = s.to(self.device)
            a = a.to(self.device)
    #---------------------------------------------------------------------------------        
            # rs, s1s, ds = self.ps.predict(s, (0.2*a).tanh())     # for UAV
   #---------------------------------------------------------------------------------           
            rs, s1s, ds = self.ps.predict(s, a)
            return rs.detach(), s1s.detach(), ds.detach(), s.detach()
    
    def load_model(self, pretrained_model):
        dic = torch.load(pretrained_model)
        self.load_state_dict(dic[''])



class HiddenAgent(ModelBasedAgent):
    def __init__(self, logger, device, agent_args, **kwargs):
        super().__init__(logger, device, agent_args, **kwargs)
        self.hidden_state_dim = agent_args.hidden_state_dim
        self.embedding_sizes = agent_args.embedding_sizes
        self.embedding_layers = self._init_embedding_layers()
        self.optimizer_p.add_param_group({'params': self.embedding_layers.parameters()})
    
    def act(self, s, requires_log=False):
        s = s.detach()
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s).detach()
        return super().act(s, requires_log)
    
    def get_logp(self, s, a):
        s = s.detach()
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s).detach()
        return super().get_logp(s, a)
    
    def updateModel(self, s, a, r, s1, d):
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s)
        if s1.size()[-1] != self.hidden_state_dim:
            s1 = self._state_embedding(s1)
        return super().updateModel(s, a, r, s1, d)
    
    def model_step(self, s, a):
        if s.size()[-1] != self.hidden_state_dim:
            s = self._state_embedding(s)
        return super().model_step(s, a)

    def _init_embedding_layers(self):
        embedding_layers = nn.ModuleList()
        for _ in range(self.n_agent):
            embedding_layers.append(MLP(self.embedding_sizes, activation=nn.ReLU))
        return embedding_layers.to(self.device)
    
    def _state_embedding(self, s):
        embeddings = []
        for i in range(self.n_agent):
            embeddings.append(self.embedding_layers[i](s.select(dim=-2, index=i).to(self.device)))
        embeddings = torch.stack(embeddings, dim=-2)
        return embeddings

class DMPOAgent(ModelBasedAgent, DPPOAgent):
    def __init__(self, logger, device, agent_args,env_args, **kwargs):
        super().__init__(logger, device, agent_args,env_args, **kwargs)
    
    def checkConverged(self, ls_info):
        rs = [info[0] for info in ls_info]
        r_converged = len(rs) > 8 and np.mean(rs[-3:]) < np.mean(rs[:-5])
        entropies = [info[1] for info in ls_info]
        entropy_converged = len(entropies) > 8 and np.abs(np.mean(entropies[-3:]) / np.mean(entropies[:-5]) - 1) < 1e-2
        kls = [info[2] for info in ls_info]
        kl_exceeded = False
        if self.target_kl is not None:
            kls = [kl > 1.5 * self.target_kl for kl in kls]
            kl_exceeded = any(kls)
        return kl_exceeded or r_converged and entropy_converged

class MB_DPPOAgent_Hidden(HiddenAgent, DMPOAgent):
    def __init__(self, logger, device, agent_args, **kwargs):
        super().__init__(logger, device, agent_args, **kwargs)
    
    def checkConverged(self, ls_info):
        rs = [info[0] for info in ls_info]
        r_converged = len(rs) > 8 and np.mean(rs[-3:]) < np.mean(rs[:-5])
        entropies = [info[1] for info in ls_info]
        entropy_converged = len(entropies) > 8 and np.abs(np.mean(entropies[-3:]) / np.mean(entropies[:-5]) - 1) < 1e-2
        kls = [info[2] for info in ls_info]
        kl_exceeded = False
        if self.target_kl is not None:
            kls = [kl > 1.5 * self.target_kl for kl in kls]
            kl_exceeded = any(kls)
        return kl_exceeded or r_converged and entropy_converged

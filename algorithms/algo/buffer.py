# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 01:56:13 2022

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
#from ray.state import actors
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
# import torch.multiprocessing as mp
from torch import distributed as dist
import argparse


class MultiCollect:
    def __init__(self, adjacency, device='cuda'):
        """
        Method: 'gather', 'reduce_mean', 'reduce_sum'.
        Adjacency: torch Tensor.
        Everything outward would be in the same device specifed in the initialization parameter.
        """
        self.device = device
        n = adjacency.size()[0]
        adjacency = adjacency > 0 # Adjacency Matrix, with size n_agent*n_agent. 
        adjacency = adjacency | torch.eye(n, device=device).bool() # Should contain self-loop, because an agent should utilize its own info.
        adjacency = adjacency.to(device)
        # print('a=',adjacency)
        self.degree = adjacency.sum(dim=1) # Number of information available to the agent.
        self.indices = []
        index_full = torch.arange(n, device=device)
        for i in range(n):
            self.indices.append(torch.masked_select(index_full, adjacency[i])) # Which agents are needed.
        


    def gather(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim_i] for i in range(n_agent)]
        """
        return self._collect('gather', tensor)

    def reduce_mean(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim] for i in range(n_agent)]
        """
        return self._collect('reduce_mean', tensor)

    def reduce_sum(self, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: [[batch_size, dim] for i in range(n_agent)]
        """
        return self._collect('reduce_sum', tensor)

    def _collect(self, method, tensor):
        """
        Input shape: [batch_size, n_agent, dim]
        Return shape: 
            gather: [[batch_size, dim_i] for i in range(n_agent)]
            reduce: [batch_size, n_agent, dim]  # same as input
        """
        tensor = tensor.to(self.device)
        if len(tensor.shape) == 1:
            tensor = tensor.unsqueeze(0)
        if len(tensor.shape) == 2:
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        result = []
        for i in range(n):
            if method == 'gather':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).view(b, -1))
            elif method == 'reduce_mean':
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).mean(dim=1))
            else:
                result.append(torch.index_select(tensor, dim=1, index=self.indices[i]).sum(dim=1))
        if method != 'gather':
            result = torch.stack(result, dim=1)
        return result

class Trajectory:
    def __init__(self, **kwargs):
        """
        Data are of size [T, n_agent, dim].
        """
        self.names = ["s", "a", "r", "s1", "d", "logp"]
        self.dict = {name: kwargs[name] for name in self.names}
        self.length = self.dict["s"].size()[0]
          
    def getFraction(self, length, start=None):

        if self.length < length:
            length = self.length
        start_max = self.length - length
        if start is None:
            start = torch.randint(low=0, high=start_max+1, size=(1,)).item()
            
        start = min(max(start, 0), start_max) 
        
        # if start > start_max:
        #     start = start_max
        # if start < 0:
        #     start = 0
      
        new_dict = {name: self.dict[name][start:start+length] for name in self.names}
        return Trajectory(**new_dict)
    
    def __getitem__(self, key):
        assert key in self.names
        return self.dict[key]
    
    @classmethod
    def names(cls):
        return ["s", "a", "r", "s1", "d", "logp"]

class TrajectoryBuffer:
    def __init__(self, device="cuda"):
        self.device = device
        self.s, self.a, self.r, self.s1, self.d, self.logp = [], [], [], [], [], []
    
    def store(self, s, a, r, s1, d, logp):
        """
        Would be converted into [batch_size, n_agent, dim].
        """
        device = self.device
        [s, r, s1, logp] = [torch.as_tensor(item, device=device, dtype=torch.float) for item in [s, r, s1, logp]]
        d = torch.as_tensor(d, device=device, dtype=torch.bool)
        a = torch.as_tensor(a, device=device)
        while s.dim() <= 2:
            s = s.unsqueeze(dim=0)
        b, n, dim = s.size()
        
        if d.dim() <= 1:
            d = d.unsqueeze(0)
        d = d[:, :n]
        if r.dim() <= 1:
            r = r.unsqueeze(0)
        r = r[:, :n]
        [s, a, r, s1, d, logp] = [item.view(b, n, -1) for item in [s, a, r, s1, d, logp]]
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.s1.append(s1)
        self.d.append(d)
        self.logp.append(logp)
    
    def retrieve(self, length=None):
        """
        Returns trajectories with s, a, r, s1, d, logp.
        Data are of size [T, n_agent, dim]
        """
        names = ["s", "a", "r", "s1", "d", "logp"]
        trajs = []
        traj_all = {}
        if self.s == []:
            return []
        for name in names:
            traj_all[name] = torch.stack(self.__getattribute__(name), dim=1)
        n = traj_all['s'].size()[0]
        for i in range(n):
            traj_dict = {}
            for name in names:
                traj_dict[name] = traj_all[name][i]  #ndecth batch into single traj
            trajs.append(Trajectory(**traj_dict))
        return trajs

class ModelBuffer:
    def __init__(self, max_traj_num):
        self.max_traj_num = max_traj_num
        self.trajectories = []
        self.ptr = -1
        self.count = 0
    
    def storeTraj(self, traj):
        if self.count < self.max_traj_num:
            self.trajectories.append(traj)
            self.ptr = (self.ptr + 1) % self.max_traj_num
            self.count = min(self.count + 1, self.max_traj_num)
        else:
            self.trajectories[self.ptr] = traj
            self.ptr = (self.ptr + 1) % self.max_traj_num
    
    def storeTrajs(self, trajs):
        for traj in trajs:
            self.storeTraj(traj)
    
    def sampleTrajs(self, n_traj):
        traj_idxs = np.random.choice(range(self.count), size=(n_traj,), replace=True)
        return [self.trajectories[i] for i in traj_idxs]
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 02:47:58 2022

@author: 86153
"""


############             Custom Environments     #######################################

from gym.spaces import Box, Discrete
import torch
import math
from math import sqrt, pow
import numpy as np
import random
import gym
from gym import spaces
from gym.utils import seeding
import time



class Custom_Env(gym.Env):

    def __init__(self):
        
         # state limit
        self.min_position = -5000
        self.max_position = 5000


        self.low_state = np.array(
            [self.min_position, self.min_position], dtype=np.float32
        )
        
        self.high_state = np.array(
            [self.max_position, self.max_position], dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low = self.low_state,
            high = self.high_state,
            dtype = np.float32
        )  
        self.action_space = Discrete(5)

        
        self.n_agent = 25
        self.n_s_ls, self.n_a_ls, self.coop_gamma, self.distance_mask, self.neighbor_mask \
            = [], [], -1, np.zeros((self.n_agent, self.n_agent)), np.zeros((self.n_agent, self.n_agent))
            

        self.init_neighbor_mask()  
        self.init_distance_mask() 
        self.seed()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        '''update state '''        

        return np.array(self.state), np.array(reward), np.array(done), np.array({})

    def get_state_(self):
        
        return State
  
    def reset(self):
        
        '''reset state'''
        
        return np.array(self.state)


    def init_neighbor_mask(self):
        n = self.n_agent
        for i in range(n):
            self.neighbor_mask[i][i] = 1
            self.neighbor_mask[i][(i+1)%n] = 1
            self.neighbor_mask[i][(i+n-1)%n] = 1

    def init_distance_mask(self):
        n = self.n_agent
        for i in range(n):
            for j in range(n):
                self.distance_mask[i][j] = min((i-j+n)%n, (j-i+n)%n)



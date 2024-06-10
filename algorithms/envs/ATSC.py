# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 19:00:22 2022

@author: 86153
"""


import gym
import numpy as np
from .NCS.large_grid_env import LargeGridEnv
from .NCS.real_net_env import RealNetEnv
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
# from .NCS.envs.large_grid_data.build_file import main
# main()
# from ..utils import listStack




# def Grid_Env():
#     # return GridWrapper('NCS/config/config_ma2c_nc_grid.ini', bias=0, std=100)
#     config = configparser.ConfigParser()
#     # config.read('D:/A_RL/MB-MARL/algorithms/envs/NCS/config/config_ma2c_nc_grid.ini')
#     config.read('algorithms/envs/NCS/config/config_ma2c_nc_grid.ini')
#     env = LargeGridEnv(config['ENV_CONFIG'])  
#     return env

# # def Monaco_Env():
# #     # return GridWrapper('NCS/config/config_ma2c_nc_grid.ini', bias=0, std=100)
# #     config = configparser.ConfigParser()
# #     config.read('algorithms/envs/NCS/config/config_ma2c_nc_net.ini')
# #     env = RealNetEnv(config['ENV_CONFIG'])  
# #     return env





class ATSCWrapper(gym.Wrapper):
    def __init__(self, config_path, n_agent,):
        # k-hop
        self.n_agent = n_agent
        config_path = os.path.join(os.path.dirname("."), config_path)
        config = configparser.ConfigParser()
        config.read(config_path)
        config = config['ENV_CONFIG']
        if self.n_agent == 28:
            env = RealNetEnv(config)
            
            # print('aaaaa=',env.phase_node_map)
            
            phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
            
            # print('aaaaa=',self.n_action)
        else:
            env = LargeGridEnv(config)
            # phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [5]*25
            # self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
            #print('aaaaa=',self.n_action)
        super().__init__(env)
        
    def reset(self):
        state = self.env.reset()
        if self.n_agent == 25:
            #state = np.array(state, dtype=np.float32)
            if any(isinstance(i, (list, tuple, np.ndarray)) for i in state):
                state = state
            else:
                state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        self.state = state
        # print('1111=',state)
        return state    
    
    # def rescaleReward(self, reward, _):
    #     return reward*200/720*self.n_agent
        
    def step(self, action):
        # print('1111=',action)
        """
        reward scaling is necessary since SAC temperature tuning can be slow to adapt to large reward
        """
        if self.n_agent == 28:
            for i in range(len(action)):
                if action[i]>= self.n_action[i]:
                    
                    if self.n_action[i] == 2:
                        if action[i] == 2 or action[i] == 3:
                            action[i] = 0
                        else:
                            action[i] = 1
                            
                    elif self.n_action[i] == 3:
                        if action[i] == 3:
                            action[i] = 0
                        elif action[i] == 4:
                            action[i] = 1
                        elif action[i] == 5:
                            action[i] = 2

                    elif self.n_action[i] == 4:
                        if action[i] == 4:
                            action[i] = 0
                        elif action[i] == 5:
                            action[i] = 1

                    elif self.n_action[i] == 5:
                        if action[i] == 5:
                            action[i] = 4

                
                    # action[i] = np.random.randint(self.n_action[i])
                    
        # print('2222=',action)
                
        state, reward, done, info = self.env.step(action)
        
        #print('ddddd=',done)
        
        # for i in range(28):
        #     print('sssssssssssssssssssssss')
        #     print('aaaaaaaa=',state[i].shape)
        
        
        if self.n_agent == 25:
            #state = np.array(state, dtype=np.float32)
            if any(isinstance(i, (list, tuple, np.ndarray)) for i in state):
                state = state
            else:
                state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        reward = np.array(reward, dtype=np.float32)
        done = np.array([done]*self.n_agent, dtype=np.float32)
        # print('dddd=',done)
        # done = np.array(done, dtype=np.float32)
        self.state=state
        return state, reward/720, done, None

    def get_state_(self):
        return self.state
    
def Grid_Env():
    return ATSCWrapper("algorithms/envs/NCS/config/config_ma2c_nc_grid.ini", 25)

def Monaco_Env():
    return ATSCWrapper("algorithms/envs/NCS/config/config_ma2c_nc_net.ini", 28)






class ATSCWrapper_2(gym.Wrapper):
    def __init__(self, env_config, n_agent,):
        # k-hop
        self.n_agent = n_agent
        self.n_agents = n_agent
        
        #print("n_agent=",n_agent)

        if self.n_agent == 28:
            env = RealNetEnv(env_config)
            
            # print('aaaaa=',env.phase_node_map)
            
            phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
            
            # print('aaaaa=',self.n_action)
        else:
            env = LargeGridEnv(env_config)
            # phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [5]*25
            # self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
            #print('aaaaa=',self.n_action)
        super().__init__(env)
        
    def reset(self):
        state = self.env.reset()
        #print("state=",len(state))
        if self.n_agent == 25:
            state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        self.state = state
        # print('1111=',state)
        return state    
    
    # def rescaleReward(self, reward, _):
    #     return reward*200/720*self.n_agent
        
    def step(self, action):
        # print('1111=',action)
        """
        reward scaling is necessary since SAC temperature tuning can be slow to adapt to large reward
        """
        if self.n_agent == 28:
            for i in range(len(action)):
                if action[i]>= self.n_action[i]:
                    
                    if self.n_action[i] == 2:
                        if action[i] == 2 or action[i] == 3:
                            action[i] = 0
                        else:
                            action[i] = 1
                            
                    elif self.n_action[i] == 3:
                        if action[i] == 3:
                            action[i] = 0
                        elif action[i] == 4:
                            action[i] = 1
                        elif action[i] == 5:
                            action[i] = 2

                    elif self.n_action[i] == 4:
                        if action[i] == 4:
                            action[i] = 0
                        elif action[i] == 5:
                            action[i] = 1

                    elif self.n_action[i] == 5:
                        if action[i] == 5:
                            action[i] = 4

                
                    # action[i] = np.random.randint(self.n_action[i])
                    
        # print('2222=',action)
                
        state, reward, done, info = self.env.step(action)
        
        #print('ddddd=',done)
        
        # for i in range(28):
        #     print('sssssssssssssssssssssss')
        #     print('aaaaaaaa=',state[i].shape)
        
        
        if self.n_agent == 25:
            state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        reward = np.array(reward, dtype=np.float32)
        done = np.array([done]*self.n_agent, dtype=np.float32)
        # print('dddd=',done)
        # done = np.array(done, dtype=np.float32)
        self.state=state
        return state, reward/720, done, None

    def get_state_(self):
        return self.state
    
def Grid_Env_2(config_path):
    return ATSCWrapper_2(config_path, 25)

def Monaco_Env_2(config_path):
    return ATSCWrapper_2(config_path, 28)


class ATSCWrapper_3(gym.Wrapper):
    def __init__(self, env_config, n_agent,):
        # k-hop
        self.n_agent = n_agent
        self.n_agents = n_agent



        self.max_time_steps = 720


        if self.n_agent == 28:
            env = RealNetEnv(env_config)
            
            # print('aaaaa=',env.phase_node_map)
            
            phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
            self.n_actions = 6
            self.n_obs = 22
            
            # print('aaaaa=',self.n_action)
        else:
            env = LargeGridEnv(env_config)
            # phases = [env.phase_node_map[node] for node in env.node_names]
            self.n_action = [5]*25
            self.n_actions = 5
            self.n_obs = 12
            # self.n_action = [env.phase_map.get_phase_num(item) for item in phases]
            #print('aaaaa=',self.n_action)
        super().__init__(env)
    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}        
    def reset(self):
        state = self.env.reset()
        if self.n_agent == 25:
            state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        self.state = state
        # print('1111=',state)
        #return state 
        return {i: obs for i, obs in enumerate(state)}   
    
    # def rescaleReward(self, reward, _):
    #     return reward*200/720*self.n_agent
        
    def step(self, action):
        # print('1111=',action)
        """
        reward scaling is necessary since SAC temperature tuning can be slow to adapt to large reward
        """
        if self.n_agent == 28:
            for i in range(len(action)):
                if action[i]>= self.n_action[i]:
                    
                    if self.n_action[i] == 2:
                        if action[i] == 2 or action[i] == 3:
                            action[i] = 0
                        else:
                            action[i] = 1
                            
                    elif self.n_action[i] == 3:
                        if action[i] == 3:
                            action[i] = 0
                        elif action[i] == 4:
                            action[i] = 1
                        elif action[i] == 5:
                            action[i] = 2

                    elif self.n_action[i] == 4:
                        if action[i] == 4:
                            action[i] = 0
                        elif action[i] == 5:
                            action[i] = 1

                    elif self.n_action[i] == 5:
                        if action[i] == 5:
                            action[i] = 4

                
                    # action[i] = np.random.randint(self.n_action[i])
                    
        # print('2222=',action)
                
        state, reward, done, info = self.env.step(action)
        
        #print('ddddd=',done)
        
        # for i in range(28):
        #     print('sssssssssssssssssssssss')
        #     print('aaaaaaaa=',state[i].shape)
        
        
        if self.n_agent == 25:
            state = np.array(state, dtype=np.float32)
        else:
            tmp = state
            state = np.zeros((28, 22), dtype=np.float32)
            for i in range(28):
                state[i, :len(tmp[i])] = np.array(tmp[i])
        reward = np.array(reward, dtype=np.float32)
        done = np.array([done]*self.n_agent, dtype=np.float32)
        # print('dddd=',done)
        # done = np.array(done, dtype=np.float32)
        self.state=state
        #return state, reward/720, done, None
        return self.to_dict(np.array(state, dtype=np.float32)), self.to_dict(reward/720), \
               self.to_dict(done), {"r": 000}


    def close(self):
        return

    def get_avail_agent_actions(self, handle):
        return

    def get_state_(self):
        return self.state
    def get_state(self):
        return self.state
    
def Grid_Env_3(config_path):
    return ATSCWrapper_3(config_path, 25)

def Monaco_Env_3(config_path):
    return ATSCWrapper_3(config_path, 28)
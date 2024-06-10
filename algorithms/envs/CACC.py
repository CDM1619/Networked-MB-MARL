import gym
import numpy as np
from .NCS.cacc_env import CACCEnv
from gym.spaces import Box, Discrete
import configparser
import os
import pdb
from ..utils import listStack
import torch

COLLISION_WT = 5
COLLISION_HEADWAY = 10
VDIFF = 5
class CACCWrapper(gym.Wrapper):
    def __init__(self, config_path, bias=0, std=1, test=False):
        # k-hop
        config_path = os.path.join(os.path.dirname(__file__), config_path)
        config = configparser.ConfigParser()
        config.read(config_path)
        env = CACCEnv(config['ENV_CONFIG'])
        env.init_data(True, False, "/tmp")
        super().__init__(env)
        self.observation_space = Box(-1e6, 1e6, [5])
        self.action_space = Discrete(4)
        self.bias=bias
        self.std=std
        self.test = test
        self.n_agents = 8
        env.neighbor_mask += np.eye(env.n_agent, dtype=env.neighbor_mask.dtype)
    
    def ifCollide(self):
        ob = self.state
        normalized_v = np.array([item[0] for item in ob])
        normalized_h = np.array([item[3] for item in ob])
        v = normalized_v * self.v_star + self.v_star
        v = np.concatenate((np.array([self.v_star]), v), axis=0)
        h = normalized_h * self.h_star + self.h_star
        h = h - (v[:-1]-v[1:])*self.dt
        if np.min(h) < self.h_min:
            return True
        return False
    
    def reset(self):
        state = self.env.reset()
        state = np.array(state, dtype=np.float32)
        self.state = state
        return state
    
    def get_reward_(self):
        # give large penalty for collision
        h_rewards = -(self.hs_cur - self.h_star) ** 2
        v_rewards = -self.a * (self.vs_cur - self.v_star) ** 2
        u_rewards = -self.b * (self.us_cur) ** 2
        if self.train_mode:
            c_rewards = -COLLISION_WT * (np.minimum(self.hs_cur - COLLISION_HEADWAY, 0)) ** 2
        else:
            c_rewards = 0
        rewards = h_rewards + v_rewards + u_rewards + c_rewards
       # rewards = v_rewards
        if np.min(self.hs_cur) < self.h_min:
            self.collision = True
            collided = self.hs_cur < self.h_min
            for i in range(self.hs_cur.shape[0]):
                if collided[i]:
                    rewards[i] -= self.G
        # print(rewards)
        return rewards
    
    def _comparable_reward(self):
        return self.env._get_reward()
    
    def state2Reward(self, state):
        # accepts a (gpu) tensor
        # Deprecated
        reward, done =  self.env.state2Reward(state)
        return (reward+self.bias)/self.std, done
    
    # def rescaleReward(self, acc_reward, episode_len):
    #     """
    #     acc_reward is sum over trajectory, mean over agent
    #     """
    #     acc_reward = acc_reward*self.std
    #     acc_reward -= episode_len*self.bias
    #     reward = acc_reward*8/episode_len
    #     return reward
        
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        # for action dim problem list 1 * action_dim
        if type(action[0]) == list:
            action = action[0]
        state, reward, done, info = self.env.step(action)
        if self.test:
            reward = self._comparable_reward()
        state = np.array(state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array([done]*8, dtype=np.float32)
        self.state=state
        reward = (reward+self.bias)/self.std
        
        
        #print("reward=",reward)
        return state, np.clip(reward, -5, 5), done, None
        
        #return state, reward, done, None

    def get_state(self):
        return self.state

    def get_state_(self):
        return self.state


def CACC_catchup():
    return CACCWrapper('NCS/config/config_ma2c_nc_catchup.ini', bias=0, std=100)

def CACC_slowdown():
    return CACCWrapper('NCS/config/config_ma2c_nc_slowdown.ini', bias=0, std=100)

def CACC_catchup_test():
    return CACCWrapper('NCS/config/config_ma2c_nc_catchup.ini', bias=0, std=100, test=True)

def CACC_slowdown_test():
    return CACCWrapper('NCS/config/config_ma2c_nc_slowdown.ini', bias=0, std=100, test=True)
    
    

class CACCWrapper_2(gym.Wrapper):
    def __init__(self, env_config, bias=0, std=1, test=False):
        # k-hop

        env = CACCEnv(env_config)
        env.init_data(True, False, "/tmp")
        super().__init__(env)
        self.observation_space = Box(-1e6, 1e6, [5])
        self.action_space = Discrete(4)
        self.bias=bias
        self.std=std
        self.test = test
        self.n_agents = 8
        #env.neighbor_mask += np.eye(env.n_agent, dtype=env.neighbor_mask.dtype)
    
    def ifCollide(self):
        ob = self.state
        normalized_v = np.array([item[0] for item in ob])
        normalized_h = np.array([item[3] for item in ob])
        v = normalized_v * self.v_star + self.v_star
        v = np.concatenate((np.array([self.v_star]), v), axis=0)
        h = normalized_h * self.h_star + self.h_star
        h = h - (v[:-1]-v[1:])*self.dt
        if np.min(h) < self.h_min:
            return True
        return False
    
    def reset(self, gui=False, test_ind = -1):
        state = self.env.reset()

        #state = np.array(state, dtype=np.float32)
        self.state = state
        return state
    
    def get_reward_(self):
        # give large penalty for collision
        h_rewards = -(self.hs_cur - self.h_star) ** 2
        v_rewards = -self.a * (self.vs_cur - self.v_star) ** 2
        u_rewards = -self.b * (self.us_cur) ** 2
        if self.train_mode:
            c_rewards = -COLLISION_WT * (np.minimum(self.hs_cur - COLLISION_HEADWAY, 0)) ** 2
        else:
            c_rewards = 0
        rewards = h_rewards + v_rewards + u_rewards + c_rewards
       # rewards = v_rewards
        if np.min(self.hs_cur) < self.h_min:
            self.collision = True
            collided = self.hs_cur < self.h_min
            for i in range(self.hs_cur.shape[0]):
                if collided[i]:
                    rewards[i] -= self.G
        # print(rewards)
        return rewards
    
    def _comparable_reward(self):
        return self.env._get_reward()
    
    def state2Reward(self, state):
        # accepts a (gpu) tensor
        # Deprecated
        reward, done =  self.env.state2Reward(state)
        return (reward+self.bias)/self.std, done
    
    # def rescaleReward(self, acc_reward, episode_len):
    #     """
    #     acc_reward is sum over trajectory, mean over agent
    #     """
    #     acc_reward = acc_reward*self.std
    #     acc_reward -= episode_len*self.bias
    #     reward = acc_reward*8/episode_len
    #     return reward
        
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        # for action dim problem list 1 * action_dim
        if type(action[0]) == list:
            action = action[0]
        state, reward, done, info = self.env.step(action)
        if self.test:
            reward = self._comparable_reward()
        #state = np.array(state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array([done]*8, dtype=np.float32)
        self.state=state
        reward = (reward+self.bias)/self.std
        #print(reward)
        return state, np.clip(reward, -5, 5), done, None

    def get_state(self):
        return self.state

    def get_state_(self):
        return self.state


def CACC_catchup_2(config_path):
    return CACCWrapper_2(config_path, bias=0, std=100)
def CACC_slowdown_2(config_path):
    return CACCWrapper_2(config_path, bias=0, std=100)



class CACCWrapper_3(gym.Wrapper):
    def __init__(self, env_config, bias=0, std=1, test=False):
        # k-hop

        self.env = CACCEnv(env_config)
        self.env.init_data(True, False, "/tmp")
        super().__init__(self.env)
        self.observation_space = Box(-1e6, 1e6, [5])
        self.action_space = Discrete(4)
        self.bias=bias
        self.std=std
        self.test = test
        self.n_agents = 8



        self.n_obs = 5
        self.n_actions = 4
        self.max_time_steps = 600


        #env.neighbor_mask += np.eye(env.n_agent, dtype=env.neighbor_mask.dtype)
    def to_dict(self, l):
        return {i: e for i, e in enumerate(l)}
            
    def ifCollide(self):
        ob = self.state
        normalized_v = np.array([item[0] for item in ob])
        normalized_h = np.array([item[3] for item in ob])
        v = normalized_v * self.v_star + self.v_star
        v = np.concatenate((np.array([self.v_star]), v), axis=0)
        h = normalized_h * self.h_star + self.h_star
        h = h - (v[:-1]-v[1:])*self.dt
        if np.min(h) < self.h_min:
            return True
        return False
    
    def reset(self, gui=False, test_ind = -1):
        state = self.env.reset()

        #state = np.array(state, dtype=np.float32)
        self.state = state
        #return state
        return {i: obs for i, obs in enumerate(state)}
    
    def get_reward_(self):
        # give large penalty for collision
        h_rewards = -(self.hs_cur - self.h_star) ** 2
        v_rewards = -self.a * (self.vs_cur - self.v_star) ** 2
        u_rewards = -self.b * (self.us_cur) ** 2
        if self.train_mode:
            c_rewards = -COLLISION_WT * (np.minimum(self.hs_cur - COLLISION_HEADWAY, 0)) ** 2
        else:
            c_rewards = 0
        rewards = h_rewards + v_rewards + u_rewards + c_rewards
       # rewards = v_rewards
        if np.min(self.hs_cur) < self.h_min:
            self.collision = True
            collided = self.hs_cur < self.h_min
            for i in range(self.hs_cur.shape[0]):
                if collided[i]:
                    rewards[i] -= self.G
        # print(rewards)
        return rewards
    
    def _comparable_reward(self):
        return self.env._get_reward()
    
    def state2Reward(self, state):
        # accepts a (gpu) tensor
        # Deprecated
        reward, done =  self.env.state2Reward(state)
        return (reward+self.bias)/self.std, done
    
    # def rescaleReward(self, acc_reward, episode_len):
    #     """
    #     acc_reward is sum over trajectory, mean over agent
    #     """
    #     acc_reward = acc_reward*self.std
    #     acc_reward -= episode_len*self.bias
    #     reward = acc_reward*8/episode_len
    #     return reward
        
    def step(self, action):
        if isinstance(action, np.ndarray):
            action = action.tolist()
        # for action dim problem list 1 * action_dim
        if type(action[0]) == list:
            action = action[0]
        state, reward, done, info = self.env.step(action)
        if self.test:
            reward = self._comparable_reward()
        #state = np.array(state, dtype=np.float32)
        reward = np.array(reward, dtype=np.float32)
        if done==0:
            done = False
        else:
            done = True
        done = np.array([done]*8, dtype=np.float32)
        self.state=state
        reward = (reward+self.bias)/self.std
        
        reward = np.clip(reward, -5, 5)

        #return state, np.clip(reward, -5, 5), done, None
        return self.to_dict(np.array(state, dtype=np.float32)), self.to_dict(reward), \
               self.to_dict(done), {"r": info}

    def close(self):
        return

    def get_avail_agent_actions(self, handle):
        return

    def get_state(self):
        return self.state

    def get_state_(self):
        return self.state


def CACC_catchup_3(config_path):
    return CACCWrapper_3(config_path, bias=0, std=100)
def CACC_slowdown_3(config_path):
    return CACCWrapper_3(config_path, bias=0, std=100)
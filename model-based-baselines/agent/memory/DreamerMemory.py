from matplotlib.style import available
import numpy as np
import torch

from environments import Env


class DreamerMemory:
    def __init__(self, config, capacity, sequence_length, action_size, obs_size, n_agents, device, env_type):
        self.config = config
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.action_size = action_size
        self.obs_size = obs_size
        self.device = device
        self.env_type = env_type
        self.init_buffer(n_agents, env_type)

    def init_buffer(self, n_agents, env_type):
        self.observations = np.empty((self.capacity, n_agents, self.obs_size), dtype=np.float32)
        self.actions = np.empty((self.capacity, n_agents, self.action_size), dtype=np.float32)
        self.av_actions = np.empty((self.capacity, n_agents, self.action_size),
                                   dtype=np.float32)
        self.rewards = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.dones = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.fake = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.last = np.empty((self.capacity, n_agents, 1), dtype=np.float32)
        self.next_idx = 0
        self.n_agents = n_agents
        self.full = False

    def append(self, obs, action, reward, done, fake, last, av_action):
        if self.actions.shape[-2] != action.shape[-2]:
            self.init_buffer(action.shape[-2], self.env_type)
        
        #print("len(obs)")
        for i in range(len(obs)):
            #print("iiiiiiii=",i)
            self.observations[self.next_idx] = obs[i]
            self.actions[self.next_idx] = action[i]
            #print("self.av_actions=",self.av_actions)
            #print("self.next_idx=",self.next_idx)
            #print("len_obs=",len(obs))
            #print("len_av_action=",len(av_action))

            if av_action is not None:
                self.av_actions[self.next_idx] = av_action[i]
            self.rewards[self.next_idx] = reward[i]
            self.dones[self.next_idx] = done[i]
            self.fake[self.next_idx] = fake[i]
            self.last[self.next_idx] = last[i]
            self.next_idx = (self.next_idx + 1) % self.capacity
            self.full = self.full or self.next_idx == 0

    def tenzorify(self, nparray):
        return torch.from_numpy(nparray).float()

    def sample(self, batch_size, repeat=True):
        return self.get_transitions(self.sample_positions(batch_size, repeat))

    def sample_all(self):
        # model training中训练过的样本拉出来作为m_r_predictor的样本 # list(set(self.sampled_idx))
        idx_splits = np.split(np.array(self.sampled_idx), self.config.m_r_predictor_epochs) # epoch个片段
        seq_idxs = [np.asarray([np.arange(start, start + self.sequence_length)%self.capacity for start in idxs], dtype=np.int16) for idxs in idx_splits]
        return [self.get_transitions(idx) for idx in seq_idxs] # len: 60, shape: (T, B, n_ags, _dim)

    def process_batch(self, val, idxs, batch_size):
        return torch.as_tensor(val[idxs].reshape(self.sequence_length, batch_size, self.n_agents, -1)).to(self.device)

    def get_transitions(self, idxs):
        batch_size = len(idxs)
        vec_idxs = idxs.transpose().reshape(-1)
        observation = self.process_batch(self.observations, vec_idxs, batch_size)[1:]
        reward = self.process_batch(self.rewards, vec_idxs, batch_size)[:-1]
        action = self.process_batch(self.actions, vec_idxs, batch_size)[:-1]
        av_action = self.process_batch(self.av_actions, vec_idxs, batch_size)[1:]
        done = self.process_batch(self.dones, vec_idxs, batch_size)[:-1]
        fake = self.process_batch(self.fake, vec_idxs, batch_size)[1:]
        last = self.process_batch(self.last, vec_idxs, batch_size)[1:]

        return {'observation': observation, 'reward': reward, 'action': action, 'done': done, 
                'fake': fake, 'last': last, 'av_action': av_action}

    def sample_position(self, repeat):
        # 首先明确一下repeat的含义：轨迹完全相同才算repeat，一部分重叠不算在内
        valid_idx = False
        max_avail_idx = self.capacity if self.full else self.next_idx - self.sequence_length
        while not valid_idx:
            idx = np.random.randint(0, max_avail_idx)
            idxs = np.arange(idx, idx + self.sequence_length) % self.capacity # batch里每个data都是一串连续的数据
            valid_idx = self.next_idx not in idxs[1:]  # Make sure data does not cross the memory index
            if not repeat:
                if idx in self.sampled_idx:
                    valid_idx = False
        self.sampled_idx.append(idx)
        return idxs

    def init_sampled_idx(self):
        self.sampled_idx = []

    def sample_positions(self, batch_size, repeat): # 相当于batch里每个data都是一串连续的数据
        # self.sampled_idx = []
        return np.asarray([self.sample_position(repeat) for _ in range(batch_size)])

    def __len__(self):
        return self.capacity if self.full else self.next_idx

    def clean(self):
        self.memory = list()
        self.position = 0

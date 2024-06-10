from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.distributions import OneHotCategorical

from agent.models.DreamerModel import DreamerModel
from networks.dreamer.action import Actor


class DreamerController:

    def __init__(self, config):
        self.args = config
        self.model = [DreamerModel(config).eval() for _ in range(config.n_elites)]
        if config.obs_as_pol_in:
            self.actor = Actor(config, config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS)
        else:
            self.actor = Actor(config, config.FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS) # NOTE
        self.expl_decay = config.EXPL_DECAY
        self.expl_noise = config.EXPL_NOISE
        self.expl_min = config.EXPL_MIN
        self.init_rnns()
        self.init_buffer()

    def receive_params(self, params):
        self.elite_idxs = params['elite_idxs']
        for i, idx in enumerate(params['elite_idxs']):
            self.model[i].load_state_dict(params['model'][idx])
        self.actor.load_state_dict(params['actor'])

    def init_buffer(self):
        self.buffer = defaultdict(list)

    def init_rnns(self):
        self.prev_rnn_state = None
        self.prev_actions = None

    def dispatch_buffer(self):
        total_buffer = {k: np.asarray(v, dtype=np.float32) for k, v in self.buffer.items()}
        last = np.zeros_like(total_buffer['done'])
        last[-1] = 1.0
        total_buffer['last'] = last
        self.init_rnns()
        self.init_buffer()
        return total_buffer

    def update_buffer(self, items):
        for k, v in items.items():
            if v is not None:
                self.buffer[k].append(v.squeeze(0).detach().clone().numpy())

    @torch.no_grad()
    def step(self, observations, avail_actions, nn_mask):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """        
        if self.args.obs_as_pol_in:
            # state = self.model(observations, self.prev_actions, self.prev_rnn_state, nn_mask)
            feats = observations
        else:
            # idx = np.random.choice(np.arange(len(self.elite_idxs)))
            state = self.model[0](observations, self.prev_actions, self.prev_rnn_state, nn_mask)
            feats = state.get_features()
        # print(11111, observations.shape, feats.shape) # (1, n_ags, _dim) (1, n_ags, _dim)
        action, pi = self.actor(feats)
        if avail_actions is not None:
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample()

        if not self.args.obs_as_pol_in:
            self.advance_rnns(state)
        self.prev_actions = action.clone()
        return action.squeeze(0).clone()

    def advance_rnns(self, state):
        self.prev_rnn_state = deepcopy(state)

    def exploration(self, action):
        """
        :param action: action to take, shape (1,)
        :return: action of the same shape passed in, augmented with some noise
        """
        for i in range(action.shape[0]):
            if np.random.uniform(0, 1) < self.expl_noise:
                index = torch.randint(0, action.shape[-1], (1, ), device=action.device)
                transformed = torch.zeros(action.shape[-1])
                transformed[index] = 1.
                action[i] = transformed
        self.expl_noise *= self.expl_decay
        self.expl_noise = max(self.expl_noise, self.expl_min)
        return action

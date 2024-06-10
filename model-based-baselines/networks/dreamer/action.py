import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import OneHotCategorical
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model


class Actor(nn.Module):
    def __init__(self, config, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()
        # self.config = config
        # self._activation = activation
        # self._rnn_input_model = self._build_rnn_input_model(in_dim, hidden_size)
        # self._cell = nn.GRU(hidden_size, hidden_size)
        # self.rnn_out_dim = hidden_size
        self.feedforward_model = build_model(in_dim, out_dim, layers, hidden_size, activation)

    def _build_rnn_input_model(self, in_dim, hidden_size):
        rnn_input_model = [nn.Linear(in_dim, hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def forward(self, x, seq=True):
        # assert len(x.shape) == 3
        # B, n_ags = x.shape[:2]
        # # print(111, x.shape) # (1, n_ags, _dim)
        # x = self._rnn_input_model(x) # (L, B, _dim)
        # # print(22, x.shape) # (1, n_ags, _dim)
        # if not seq: # 不是sequence,则第一维L=1
        #     x = x.reshape(1, B*n_ags, x.shape[-1])
        # x, self.hidden = self._cell(x, self.hidden)
        # # print(333, x.shape)
        # if not seq:
        #     x = x.reshape(B, n_ags, x.shape[-1])
        x = self.feedforward_model(x)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x
    
    def deter_action(self, x):
        x = self.feedforward_model(x)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x


class AttentionActor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size, layers, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, out_dim, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)

    def forward(self, state_features):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        x = self.feedforward_model(attn_embeds)
        action_dist = OneHotCategorical(logits=x)
        action = action_dist.sample()
        return action, x

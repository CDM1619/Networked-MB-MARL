import torch.nn as nn
import torch.nn.functional as F

from networks.dreamer.utils import build_model
from networks.transformer.layers import AttentionEncoder


class Critic(nn.Module):
    def __init__(self, in_dim, hidden_size, layers=2, activation=nn.ELU):
        super().__init__()
        self.hidden_size = hidden_size
        self.layers = layers
        self.activation = activation
        self.feedforward_model = build_model(in_dim, 1, layers, hidden_size, activation)

    def forward(self, state_features, actions):
        return self.feedforward_model(state_features)


class MADDPGCritic(nn.Module):
    def __init__(self, in_dim, hidden_size, activation=nn.ReLU):
        super().__init__()
        self.feedforward_model = build_model(hidden_size, 1, 1, hidden_size, activation)
        self._attention_stack = AttentionEncoder(1, hidden_size, hidden_size)
        self.embed = nn.Linear(in_dim, hidden_size)
        self.prior = build_model(in_dim, 1, 3, hidden_size, activation)

    def forward(self, state_features, actions):
        n_agents = state_features.shape[-2]
        batch_size = state_features.shape[:-2]
        embeds = F.relu(self.embed(state_features))
        embeds = embeds.view(-1, n_agents, embeds.shape[-1])
        attn_embeds = F.relu(self._attention_stack(embeds).view(*batch_size, n_agents, embeds.shape[-1]))
        return self.feedforward_model(attn_embeds)

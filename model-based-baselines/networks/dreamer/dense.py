import torch.distributions as td
import torch.nn as nn

from networks.dreamer.utils import build_model


class DenseModel(nn.Module):
    def __init__(self, in_dim, out_dim, layers, hidden, activation=nn.ELU):
        super().__init__()

        self.model = build_model(in_dim, out_dim, layers, hidden, activation)

    def forward(self, features):
        return self.model(features)


class DenseBinaryModel(DenseModel):
    def __init__(self, in_dim, out_dim, layers, hidden, activation=nn.ELU):
        super().__init__(in_dim, out_dim, layers, hidden, activation)

    def forward(self, features):
        dist_inputs = self.model(features)
        return td.independent.Independent(td.Bernoulli(logits=dist_inputs), 1)


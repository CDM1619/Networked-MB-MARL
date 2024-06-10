import torch.nn as nn
import torch


def build_model(in_dim, out_dim, layers, hidden, activation, normalize=lambda x: x):
    model = [normalize(nn.Linear(in_dim, hidden))]
    model += [activation()]
    for i in range(layers - 1):
        model += [normalize(nn.Linear(hidden, hidden))]
        model += [activation()]
    model += [normalize(nn.Linear(hidden, out_dim))]
    return nn.Sequential(*model)


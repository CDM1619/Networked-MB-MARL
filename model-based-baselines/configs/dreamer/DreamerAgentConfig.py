from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

RSSM_STATE_MODE = 'discrete'


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()
        self.HIDDEN = 32
        self.MODEL_HIDDEN = 32
        self.EMBED = 32
        self.N_CATEGORICALS = 16
        self.N_CLASSES = 16
        self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        self.DETERMINISTIC = 32
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        self.GLOBAL_FEAT = self.FEAT + self.EMBED
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 64
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 64
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 64
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 32
        self.GAMMA = 0.99
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95
        self.IN_DIM = 30
        self.LOG_FOLDER = 'wandb/'

        self.use_wandb = True
        self.obs_as_pol_in = False
        self.use_attn = True

        self.baseline = False
        if self.baseline:
            self.obs_as_pol_in = False # 要么都True要么都False
            self.use_attn = True
        
        self.n_nets = 1
        self.n_elites = 1
        self.n_fc = 3


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]

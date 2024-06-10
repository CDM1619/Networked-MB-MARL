import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
import numpy as np
import wandb

from configs.dreamer.DreamerAgentConfig import RSSMState
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.utils import build_model
from agent.optim.utils import state_divergence_loss
# from agent.optim.loss import model_loss


def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return RSSMState(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                       for key in rssm_states[0].__dict__.keys()])

class EnsembleFC(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int, weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])  # w times x + b


class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        one_hot = class_dist.sample()
        latents = one_hot + class_dist.probs - class_dist.probs.detach()
        return logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,))


class RSSMTransition(nn.Module):
    def __init__(self, config, hidden_size=200, activation=nn.ReLU):
        super().__init__()
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRU(hidden_size, self._deter_size)
        self.config = config
        
        # NOTE 弄清楚这一块的输入输出是什么，然后替换，变成参数共享
        if config.use_attn:
            self._attention_stack = AttentionEncoder(3, hidden_size, hidden_size, dropout=0.1) # NOTE
        else:
            self._fc = build_model(hidden_size, hidden_size, config.n_fc, hidden_size, activation)

        self._rnn_input_model = self._build_rnn_input_model(config.ACTION_SIZE + self._stoch_size)
        self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, config.N_CATEGORICALS, config.N_CLASSES, self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_states, mask=None):
        batch_size, n_agents = prev_actions.shape[:2]
        stoch_input = self._rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        # print(stoch_input.shape) # (?, n_ags, hidden_dim) n跟batch有关，执行时和训练时不一样
        if self.config.use_attn:
            attn = self._attention_stack(stoch_input, mask=mask) # NOTE
        else:
            attn = self._fc(stoch_input)
        # print(1111, attn.shape) # (?, n_ags, hidden_dim)
        deter_state = self._cell(attn.reshape(1, batch_size * n_agents, -1),
                                 prev_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1)
        logits, stoch_state = self._stochastic_prior_model(deter_state) # NOTE
        # print(deter_state.shape, stoch_state.shape, logits.shape) # 前两个(?, n_ags, hidden_dim)后一个(?, n_ags, n_cate*n_class)
        return RSSMState(logits=logits, stoch=stoch_state, deter=deter_state)

    def para_predict(self, prev_actions, prev_states, mask=None):
        n_trajs, batch_size, n_agents = prev_actions.shape[:3]
        prev_actions = prev_actions.reshape(n_trajs * batch_size, n_agents, -1) # (n_traj*B, n_ags, _dim)
        stoch = torch.cat([prev_state.stoch for prev_state in prev_states], dim=0) # (n_traj*B, n_ags, _dim)
        deter = torch.cat([prev_state.deter for prev_state in prev_states], dim=0) # (n_traj*B, n_ags, _dim)
        stoch_input = self._rnn_input_model(torch.cat([prev_actions, stoch], dim=-1)) # (n_traj*B, n_ags, _dim)
        # print(stoch_input.shape) # (?, n_ags, hidden_dim) n跟batch有关，执行时和训练时不一样
        if self.config.use_attn:
            attn = self._attention_stack(stoch_input, mask=mask) # NOTE
        else:
            attn = self._fc(stoch_input)
        # print(1111, attn.shape) # (?, n_ags, hidden_dim)
        deter_state = self._cell(attn.reshape(1, n_trajs * batch_size * n_agents, -1),
                                 deter.reshape(1, n_trajs * batch_size * n_agents, -1))[0]
        deter_state = deter_state.reshape(n_trajs * batch_size, n_agents, -1)
        logits, stoch_state = self._stochastic_prior_model(deter_state) # NOTE
        logits, stoch_state, deter_state = logits.reshape(n_trajs, batch_size, n_agents, logits.shape[-1]), \
                                    stoch_state.reshape(n_trajs, batch_size, n_agents, stoch_state.shape[-1]),\
                                    deter_state.reshape(n_trajs, batch_size, n_agents, deter_state.shape[-1])
        return [RSSMState(logits=logits[i], stoch=stoch_state[i], deter=deter_state[i]) for i in range(n_trajs)], logits, stoch_state, deter_state


class RSSMRepresentation(nn.Module):
    def __init__(self, config, transition_model: RSSMTransition):
        super().__init__()
        self._transition_model = transition_model
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._stochastic_posterior_model = DiscreteLatentDist(self._deter_size + config.EMBED, config.N_CATEGORICALS,
                                                              config.N_CLASSES, config.HIDDEN)

    def initial_state(self, batch_size, n_agents, **kwargs):
        return RSSMState(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def forward(self, obs_embed, prev_actions, prev_states, mask=None):
        """
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_states = self._transition_model(prev_actions, prev_states, mask) # z
        x = torch.cat([prior_states.deter, obs_embed], dim=-1)
        logits, stoch_state = self._stochastic_posterior_model(x)
        posterior_states = RSSMState(logits=logits, stoch=stoch_state, deter=prior_states.deter)
        return prior_states, posterior_states


def rollout_representation(representation_model, steps, obs_embed, action, prev_states, done):
    """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, n_agents, embedding_size)
        :param action: size(time_steps, batch_size, n_agents, action_size)
        :param prev_states: RSSM state, size(batch_size, n_agents, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, n_agents, state_size)
        """
    priors = []
    posteriors = []
    for t in range(steps):
        prior_states, posterior_states = representation_model(obs_embed[t], action[t], prev_states)
        prev_states = posterior_states.map(lambda x: x * (1.0 - done[t]))
        priors.append(prior_states)
        posteriors.append(posterior_states)

    prior = stack_states(priors, dim=0)
    post = stack_states(posteriors, dim=0)
    return prior.map(lambda x: x[:-1]), post.map(lambda x: x[:-1]), post.deter[1:]


def rollout_policy(m_r_predictor, obs_decoder, transition_model, av_action, steps, policy, prev_state, config):
    """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    state = prev_state
    next_states = []
    actions = []
    av_actions = []
    policies = []
    imag_obs = []
    minlosses, ranlosses = [], []
    for t in range(steps):
        feat = state.get_features().detach() # (B, n_ags, _dim)
        # if t == 0: # initialize rnn hidden state
        #     policy.hidden = torch.zeros(1, int(feat.shape[0]*feat.shape[1]), policy.rnn_out_dim, device=feat.device)
        if config.obs_as_pol_in:
            obs = obs_decoder(feat)[0].detach()
            imag_obs.append(obs)
            action, pi = policy(obs, seq=False)
        else:
            # obs = obs_decoder(feat)[0].detach()
            action, pi = policy(feat)
        if av_action is not None:
            avail_actions = av_action(feat).sample()
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(avail_actions.squeeze(0))
        next_states.append(state)
        policies.append(pi)
        actions.append(action)
        if config.use_MPCmodel:
            if config.use_epsilon_MPC and np.random.uniform(low=0, high=1, size=1)[0] < config.MPCepsilon:
                state = transition_model(action, state) # 只有prior，这次没有posterior了，所以m_r_predictor的输入只能是prior
            else:
                with torch.no_grad():
                    state, minloss, ranloss = MPCPredict(policy, m_r_predictor, action, state, transition_model, config, av_action=av_action)
                    minlosses.append(minloss)
                    ranlosses.append(ranloss)
        else:
            state = transition_model(action, state)
    minlosses, ranlosses = np.array(minlosses), np.array(ranlosses)
    for i in reversed(range(len(minlosses))):
        minlosses[i], ranlosses[i] = minlosses[:i+1].sum(), ranlosses[:i+1].sum()
    print('minlosses: {}'.format(np.around(np.array(minlosses), decimals=2)))
    print('ranlosses: {}'.format(np.around(np.array(ranlosses), decimals=2)))
    return {"imag_states": stack_states(next_states, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0),
            "imag_obs": torch.stack(imag_obs, dim=0) if len(imag_obs) > 0 else None}


def MPCPredict(policy, m_r_predictor, action, state, transition_model, config, av_action):
    onehot = OneHot(out_dim=action.shape[-1])
    batch_size, n_agents = action.shape[:2]
    traj_states = [state] * config.n_trajs
    action = action.unsqueeze(0).repeat(config.n_trajs, 1, 1, 1) # (n_trajs, B, n_ags, _dim)
    # traj_losses = np.zeros((config.n_trajs, batch_size))
    traj_losses = np.zeros((config.n_trajs))
    for t in range(config.MPCHorizon):
        if t == 0:
            traj_states, logits, stoch_state, deter_state = transition_model.para_predict(action, traj_states) # len: n_trajs; shape: (B, n_ags, _dim)
            first_pred = traj_states # 一个state的列表
        else:
            traj_states = transition_model.para_predict(action, traj_states)[0] # len: n_trajs; shape: (B, n_ags, _dim)
        feat = torch.stack([state.get_features() for state in traj_states])
        feat = feat.reshape(-1, n_agents, feat.shape[-1]) # (n_trajs, B, n_ags, _dim)
        loss = m_r_predictor(feat).reshape(-1, batch_size, n_agents, 1) # (n_trajs, B, n_ags, 1)
        if config.discount_MPC:
            loss *= config.MPCgamma**t
        traj_losses += loss.mean(dim=(1, 2, 3)).cpu().numpy()
        action, pi = policy(feat)
        if av_action is not None:
            avail_actions = av_action(feat).sample()
            pi[avail_actions == 0] = -1e10
            if config.DeterPolForMo:
                action = onehot.transform(pi.argmax(dim=-1, keepdim=True)).reshape(config.n_trajs, batch_size, n_agents, action.shape[-1])
            else:
                action_dist = OneHotCategorical(logits=pi)
                action = action_dist.sample().squeeze(0).reshape(config.n_trajs, batch_size, n_agents, action.shape[-1])

    best_traj = traj_losses.argmin() # (B,) TODO弄个(1, B) 代入logits那些
    # if config.use_wandb:
    #     wandb.log({'m_r_loss': traj_losses.min()})
    # idx = [[best_traj[i], i] for i in range(batch_size)]
    # best_logits, best_stoch_state, best_deter_state = logits.mean(0), stoch_state.mean(0), deter_state.mean(0) # (B, n_ags, _dim)
    # for d in range(batch_size):
    #     best_logits[d] = logits[best_traj[d].astype(np.long), d]
    next_state = first_pred[best_traj]
    # next_state = RSSMState(logits=best_logits, stoch=best_stoch_state, deter=best_deter_state)
    return next_state, traj_losses.min(), np.random.choice(traj_losses, 1)[0]


class Transform:
    def transform(self, tensor):
        raise NotImplementedError

    def infer_output_info(self, vshape_in, dtype_in):
        raise NotImplementedError

class OneHot(Transform):
    def __init__(self, out_dim):
        self.out_dim = out_dim

    def transform(self, tensor):
        y_onehot = tensor.new(*tensor.shape[:-1], self.out_dim).zero_()
        y_onehot.scatter_(-1, tensor.long(), 1)
        return y_onehot.float()

    def infer_output_info(self, vshape_in, dtype_in):
        return (self.out_dim,), torch.float32
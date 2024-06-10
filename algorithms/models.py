from re import S
from algorithms.envs.flow import networks
from math import log
import numpy as np
import ipdb as pdb
import itertools
from gym.spaces import Box, Discrete
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.categorical import Categorical
from torch.optim import Adam

# from .base_util import batch_to_seq, init_layer, one_hot


def MLP(sizes, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

def MLP_for_heterogeneous_space(heterogeneous_state_dim, sizes, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    #print("heterogeneous_state_dim=",heterogeneous_state_dim)
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(int(heterogeneous_state_dim), sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def CNN(sizes, kernels, strides, paddings, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Conv2d(sizes[j], sizes[j + 1], kernels[j], strides[j], paddings[j]), act()]
    return nn.Sequential(*layers)


class ParameterizedModel(nn.Module):
    """
        assumes parameterized state representation
        we may use a gaussian prediciton,
        but it degenrates without a kl hyperparam
        unlike the critic and the actor class, 
        the sizes argument does not include the dim of the state
        n_embedding is the number of embedding modules needed, = the number of discrete action spaces used as input
    """

    def __init__(self, env, logger, n_embedding=1, to_predict="srd", gaussian=False, **net_args):
        super().__init__()
        self.logger = logger.child("p")
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        input_dim = net_args['sizes'][0]
        output_dim = net_args['sizes'][-1]
        self.n_embedding = n_embedding
        if isinstance(self.action_space, Discrete):
            self.action_embedding = nn.Embedding(self.action_space.n, input_dim // n_embedding)
        self.net = MLP(**net_args)
        self.state_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.reward_head = nn.Linear(output_dim, 1)
        self.done_head = nn.Linear(output_dim, 1)
        self.variance_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.NLL = nn.GaussianNLLLoss(reduction='none')
        self.to_predict = to_predict
        self.gaussian = gaussian

    def forward(self, s, a, r=None, s1=None, d=None):
        embedding = s
        if isinstance(self.action_space, Discrete):
            batch_size, _ = a.shape
            action_embedding = self.action_embedding(a).view(batch_size, -1)
            embedding = embedding + action_embedding
        embedding = self.net(embedding)
        state = self.state_head(embedding)
        state_size = state.size()
        reward = self.reward_head(embedding).squeeze(1)
        if self.gaussian:
            variance = self.variance_head(embedding)
            sq_variance = variance ** 2

        if r is None:  # inference
            with torch.no_grad():
                done = torch.sigmoid(self.done_head(embedding))
                done = torch.cat([1 - done, done], dim=1)
                done = Categorical(done).sample()  # [b]
                if self.gaussian:
                    state = torch.normal(state, sq_variance)
                return reward, state, done

        else:  # training
            done = self.done_head(embedding).squeeze(1)
            if not self.gaussian:
                state_loss = self.MSE(state, s1)
                state_loss = state_loss.mean(dim=1)
                state_var = self.MSE(s1, s1.mean(dim=0, keepdim=True).expand(*s1.shape))
                # we assume the components of state are of similar magnitude
                rel_state_loss = state_loss.mean() / (state_var.mean() + 1e-5)
                self.logger.log(rel_state_loss=rel_state_loss)
            else:
                state_loss = self.NLL(state, s1, sq_variance)
                if state_loss.dim() > 1:
                    state_loss = state_loss.mean(dim=1)
                self.logger.log(state_nll_loss=state_loss, var_mean=sq_variance.mean())

            loss = state_loss
            if 'r' in self.to_predict:
                reward_loss = self.MSE(reward, r)
                loss = loss + reward_loss
                reward_var = self.MSE(reward, reward.mean(dim=0, keepdim=True).expand(*reward.shape)).mean()

                self.logger.log(reward_loss=reward_loss,
                                reward_var=reward_var,
                                reward=r)

            if 'd' in self.to_predict:
                done_loss = self.BCE(done, d)
                loss = loss + 10 * done_loss
                done = done > 0
                done_true_positive = (done * d).mean()
                d = d.mean()
                self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)

            return (loss, state.detach())


class ParameterizedModel_New(nn.Module):
    """
        assumes parameterized state representation
        we may use a gaussian prediciton,
        but it degenrates without a kl hyperparam
        unlike the critic and the actor class, 
        the sizes argument does not include the dim of the state
        n_embedding is the number of embedding modules needed, = the number of discrete action spaces used as input
    """

    def __init__(self, env, logger, n_embedding=1, to_predict="srd", gaussian=False, **net_args):
        super().__init__()
        self.logger = logger.child("p")
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.n_embedding = n_embedding
        if isinstance(self.action_space, Discrete):
            self.action_embedding = nn.Embedding(self.action_space.n, n_embedding)
        output_dim = net_args['sizes'][-1]
        """input_dim = net_args['sizes'][0]
        output_dim = net_args['sizes'][-1]
        self.n_embedding = n_embedding
        if isinstance(self.action_space, Discrete):
            self.action_embedding = nn.Embedding(self.action_space.n, input_dim // n_embedding)"""
        self.net = MLP(**net_args)
        self.state_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.reward_head = nn.Linear(output_dim, 1)
        self.done_head = nn.Linear(output_dim, 1)
        self.variance_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.NLL = nn.GaussianNLLLoss(reduction='none')
        self.to_predict = to_predict
        self.gaussian = gaussian

    def forward(self, s, a, r=None, s1=None, d=None):
        if isinstance(self.action_space, Discrete):
            batch_size, _ = a.shape
            action_embedding = self.action_embedding(a).view(batch_size, -1)
        embedding = torch.cat([s, action_embedding], dim=1)
        embedding = self.net(embedding)
        state = self.state_head(embedding)
        state_size = state.size()
        reward = self.reward_head(embedding).squeeze(1)
        if self.gaussian:
            variance = self.variance_head(embedding)
            sq_variance = variance ** 2

        if r is None:  # inference
            with torch.no_grad():
                done = torch.sigmoid(self.done_head(embedding))
                done = torch.cat([1 - done, done], dim=1)
                done = Categorical(done).sample()  # [b]
                if self.gaussian:
                    state = torch.normal(state, sq_variance)
                return reward, state, done

        else:  # training
            done = self.done_head(embedding).squeeze(1)
            if not self.gaussian:
                state_loss = self.MSE(state, s1)
                state_loss = state_loss.mean(dim=1)
                state_var = self.MSE(s1, s1.mean(dim=0, keepdim=True).expand(*s1.shape))
                # we assume the components of state are of similar magnitude
                rel_state_loss = state_loss.mean() / state_var.mean()
                self.logger.log(rel_state_loss=rel_state_loss)
            else:
                state_loss = self.NLL(state, s1, sq_variance)
                if state_loss.dim() > 1:
                    state_loss = state_loss.mean(dim=1)
                self.logger.log(state_nll_loss=state_loss, var_mean=sq_variance.mean())

            loss = state_loss
            if 'r' in self.to_predict:
                reward_loss = self.MSE(reward, r)
                loss = loss + reward_loss
                reward_var = self.MSE(reward, reward.mean(dim=0, keepdim=True).expand(*reward.shape)).mean()

                self.logger.log(reward_loss=reward_loss,
                                reward_var=reward_var,
                                reward=r)

            if 'd' in self.to_predict:
                done_loss = self.BCE(done, d)
                loss = loss + 10 * done_loss
                done = done > 0
                done_true_positive = (done * d).mean()
                d = d.mean()
                self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)

            return (loss, state.detach())

class ParameterizedModel_MBPPO(nn.Module):
    def __init__(self, logger, action_space, observation_space, n_embedding=1, to_predict="srd", gaussian=False, **net_args):
        super().__init__()
        self.logger = logger.child("p")
        self.action_space = action_space
        self.observation_space = observation_space
        self.n_embedding = n_embedding
        if isinstance(self.action_space, Discrete):
            self.action_embedding = nn.Embedding(self.action_space.n, n_embedding)
        output_dim = net_args['sizes'][-1]
        self.net = MLP(**net_args)
        self.state_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.reward_head = nn.Linear(output_dim, 1)
        self.done_head = nn.Linear(output_dim, 1)
        self.variance_head = nn.Linear(output_dim, self.observation_space.shape[0])
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.NLL = NLLLoss
        self.to_predict = to_predict
        self.gaussian = gaussian

    def forward(self, s, a, r=None, s1=None, d=None):
        """
        input shape: [batch_size, dim]
        """
        if isinstance(self.action_space, Discrete):
            batch_size, _ = a.shape
            action_embedding = self.action_embedding(a).view(batch_size, -1)
        embedding = torch.cat([s, action_embedding], dim=-1)
        embedding = self.net(embedding)
        state = self.state_head(embedding)
        state_size = state.size()
        reward = self.reward_head(embedding).squeeze(-1)
        if self.gaussian:
            variance = self.variance_head(embedding)
            sq_variance = variance ** 2

        if r is None:  # inference
            with torch.no_grad():
                done = torch.sigmoid(self.done_head(embedding))
                done = torch.cat([1 - done, done], dim=1)
                done = Categorical(done).sample()  # [b]
                if self.gaussian:
                    state = torch.normal(state, sq_variance)
                return reward, state, done

        else:  # training
            done = self.done_head(embedding).squeeze(1)
            if not self.gaussian:
                state_loss = self.MSE(state, s1)
                state_loss = state_loss.mean(dim=-1)
                state_var = self.MSE(s1, s1.mean(dim=0, keepdim=True).expand(*s1.shape))
                # we assume the components of state are of similar magnitude
                rel_state_loss = state_loss.mean() / state_var.mean()
                self.logger.log(rel_state_loss=rel_state_loss)
            else:
                state_loss = self.NLL(state, s1, sq_variance)
                if state_loss.dim() > 1:
                    state_loss = state_loss.mean(dim=-1)
                self.logger.log(state_nll_loss=state_loss, var_mean=sq_variance.mean(), state_norm=torch.norm(s))

            loss = state_loss
            if 'r' in self.to_predict:
                reward_loss = self.MSE(reward, r)
                loss = loss + reward_loss
                reward_var = self.MSE(reward, reward.mean(dim=0, keepdim=True).expand(*reward.shape)).mean()

                self.logger.log(reward_loss=reward_loss,
                                reward_var=reward_var,
                                reward=r,
                                reward_norm=torch.norm(r))

            if 'd' in self.to_predict:
                done_loss = self.BCE(done, d)
                loss = loss + 10 * done_loss
                done = done > 0
                done_true_positive = (done * d).mean()
                d = d.mean()
                self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)
            
            # debug information
            with torch.no_grad():
                debug_state_var = torch.var(s, dim=1).mean()
                debug_state_pred_mse = self.MSE(state, s1)
                debug_mse_var_ratio = debug_state_pred_mse / debug_state_var
                self.logger.log(
                    debug_state_var=debug_state_var,
                    debug_state_pred_mse=debug_state_pred_mse,
                    debug_mse_var_ratio=debug_mse_var_ratio
                )

            return (loss, state.detach())

class EnsembledModel(nn.Module):
    def __init__(self, n_p=1, *args, **kwargs):
        super().__init__()
        self.n_p = n_p
        self.ps = nn.ModuleList()
        for _ in range(n_p):
            self.ps.append(ParameterizedModel_MBPPO(*args, **kwargs))
    
    def forward(self, *args, **kwargs):
        return self.ps[np.random.randint(self.n_p)](*args, **kwargs)
    
    def train(self, s, a, r, s1, d):
        "Expect batch_size * n_p, would split the dataset for each model in the ensemble."
        batch_size = s.size()[0]
        assert batch_size >= self.n_p
        idxs = torch.from_numpy(np.random.permutation(batch_size)).to(s.device)
        split = [int(batch_size / self.n_p) * i for i in range(self.n_p + 1)]
        split[-1] = min(split[-1, batch_size])
        loss = 0
        for i in range(self.n_p):
            items = [item[split[i]:split[i+1]-1] for item in [s, a, r, s1, d]]
            loss_, _ = self.ps[i](*items)
            loss += loss_
        return loss_

class GraphConvolutionalModel(nn.Module):
    class EdgeNetwork(nn.Module):
        def __init__(self, i, j, sizes, activation=nn.ReLU, output_activation=nn.Identity):
            super().__init__()
            self.i = i
            self.j = j
            layers = []
            for t in range(len(sizes) - 1):
                act = activation if t < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[t], sizes[t + 1]), act()]
                #layers += [nn.Linear(sizes[t], sizes[t + 1])]
            self.net = nn.Sequential(*layers)
        
        def forward(self, s:torch.Tensor):
            """
            Input: [batch_size, n_agent, node_embed_dim] # raw input
            Output: [batch_size, edge_embed_dim]
            """
            s1 = s.select(dim=1, index=self.i)
            s2 = s.select(dim=1, index=self.j)
            s = torch.cat([s1, s2], dim=-1)
            return self.net(s)
    
    class NodeNetwork(nn.Module):
        def __init__(self, sizes, n_embedding=0, action_dim=0,edge_embed_dim=0, activation=nn.ReLU, output_activation=nn.ReLU):
            super().__init__()
            layers = []
            for t in range(len(sizes) - 1):
                act = activation if t < len(sizes) - 2 else output_activation
                layers += [nn.Linear(sizes[t], sizes[t + 1]), act()]
                #layers += [nn.Linear(sizes[t], sizes[t + 1])]
            self.net = nn.Sequential(*layers)
            
            
            if n_embedding != 0:
                self.action_embedding_fn = nn.Embedding(action_dim, n_embedding)
                self.action_embedding = lambda x: self.action_embedding_fn(x.squeeze(-1))
            else:
                self.action_embedding = nn.Identity()
            self.edge_embed_dim = edge_embed_dim



        def forward(self, h_ls, a):
            """
            Input: 
                h_ls: list of tensors with sizes of [batch_size, edge_embed_dim]
                a: [batch_size, action_dim]
            Output: 
                h: [batch_size, node_embed_dim]
            """

            ### embedding = 0
            ###hear 12 is edge_embed_dim
            embedding = torch.zeros([a.size()[0],self.edge_embed_dim], dtype=torch.float32, device = a.device)
            for h in h_ls:
                embedding += h
            a = self.action_embedding(a)
            while a.ndim < embedding.ndim:
                a = a.unsqueeze(-1)
            embedding = torch.cat([embedding, a], dim=-1)
            return self.net(embedding)


    class NodeWiseEmbedding(nn.Module):
        def __init__(self, n_agent, input_dim, output_dim, output_activation):
            super().__init__()
            self.nets = nn.ModuleList()
            self.n_agent = n_agent
            for _ in range(n_agent):
                self.nets.append(nn.Sequential(*[nn.Linear(input_dim, output_dim), output_activation()]))
        
        def forward(self, h):
            # input dim = 3, output the same
            items = []
            for i in range(self.n_agent):
                items.append(self.nets[i](h.select(dim=1, index=i)))
            items = torch.stack(items, dim=1)
            return items

    def __init__(self, logger, adj, state_dim, action_dim, n_agent, p_args):
        super().__init__()
        self.logger = logger.child("p")
        self.adj = adj > 0
        # print('adj=',adj)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_agent = n_agent
        self.n_conv = p_args.n_conv
        self.n_embedding = p_args.n_embedding
        self.residual = p_args.residual
        self.edge_embed_dim = p_args.edge_embed_dim
        self.edge_hidden_size = p_args.edge_hidden_size
        self.node_embed_dim = p_args.node_embed_dim
        self.node_hidden_size = p_args.node_hidden_size
        self.reward_coeff = p_args.reward_coeff

        self.node_nets = self._init_node_nets()
        self.edge_nets = self._init_edge_nets()
        self.node_embedding, self.state_head, self.reward_head, self.done_head = self._init_node_embedding()
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCELoss(reduction='none')
    
    def predict(self, s, a):
        """
            Input: 
                s: [batch_size, n_agent, state_dim]
                a: [batch_size, n_agent, action_dim]
            Output: [batch_size, n_agent, state_dim] # same as input state
        """
        with torch.no_grad():
            r1, s1, d1 = self.forward(s, a)
            done = torch.clamp(d1, 0., 1.)
            done = torch.cat([1 - done, done], dim=-1)
            done = Categorical(done).sample() > 0  # [b]
            return r1, s1, done
    
    def train(self, s, a, r, s1, d, length = 1):
       
        """
        Input shape: [batch_size, T, n_agent, dim]
        """
        pred_s, pred_r, pred_d = [], [], []
        s0 = s.select(dim=1, index=0)
        length = min(length, s.shape[1])
        
        
        
        for t in range(length):
            r_, s_, d_ = self.forward(s0, a.select(dim=1, index=t))
            pred_r.append(r_)
            pred_s.append(s_)
            pred_d.append(d_)
            s0 = s_
        reward_pred = torch.stack(pred_r, dim=1)
        state_pred = torch.stack(pred_s, dim=1)
        done_pred = torch.stack(pred_d, dim=1)

        state_loss = self.MSE(state_pred, s1).mean()  
        s1_view = s1.view(-1, s1.shape[-1])
        state_var = self.MSE(s1_view, s1_view.mean(dim=0, keepdim=True).expand(*s1_view.shape))
        rel_state_loss = state_loss / (state_var.mean() + 1e-7)
        self.logger.log(state_loss=state_loss, state_var=state_var.mean(), rel_state_loss=rel_state_loss)
        loss = state_loss

        reward_loss = self.MSE(reward_pred, r)
        loss += self.reward_coeff * reward_loss.mean()
        r_view = r.view(-1, r.shape[-1])
        reward_var = self.MSE(r_view, r_view.mean(dim=0, keepdim=True).expand(*r_view.shape)).mean()
        rel_reward_loss = reward_loss.mean() / (reward_var.mean() + 1e-7)

        self.logger.log(reward_loss=reward_loss,
                        reward_var=reward_var,
                        reward=r,
                        reward_norm=torch.norm(r),
                        rel_reward_loss=rel_reward_loss)

        d = d.float()
        done_loss = self.BCE(done_pred, d)
        loss = loss + done_loss.mean()
        done = done_pred > 0
        done_true_positive = (done * d).mean()
        d = d.mean()
        self.logger.log(done_loss=done_loss, done_true_positive=done_true_positive, done=d, rolling=100)

        return (loss, rel_state_loss)
    
    def forward(self, s, a):
        """
            Input: [batch_size, n_agent, state_dim]
            Output: [batch_size, n_agent, state_dim]
        """
        embedding = self.node_embedding(s) # dim = 3
        for _ in range(self.n_conv):
            edge_info_of_nodes = [[] for __ in range(self.n_agent)]
            for edge_net in self.edge_nets:
                edge_info = edge_net(embedding) # dim = 2
                edge_info_of_nodes[edge_net.i].append(edge_info)
                edge_info_of_nodes[edge_net.j].append(edge_info)
            node_preds = []
            for i in range(self.n_agent):
                node_net = self.node_nets[i]
                node_pred = node_net(edge_info_of_nodes[i], a.select(dim=1, index=i)) # dim = 2
                node_preds.append(node_pred)
            embedding = torch.stack(node_preds, dim=1) # dim = 3
        state_pred = self.state_head(embedding)
        if self.residual:
            state_pred += s
        reward_pred = self.reward_head(embedding)
        done_pred = self.done_head(embedding)
        return reward_pred, state_pred, done_pred

    def _init_node_nets(self):
        node_nets = nn.ModuleList()
        action_dim = self.n_embedding if self.n_embedding > 0 else self.action_dim
        sizes = [self.edge_embed_dim + action_dim] + self.node_hidden_size + [self.node_embed_dim]
        for i in range(self.n_agent):
            node_nets.append(GraphConvolutionalModel.NodeNetwork(sizes=sizes, n_embedding=self.n_embedding, action_dim=self.action_dim,edge_embed_dim=self.edge_embed_dim))
        return node_nets

    def _init_edge_nets(self):
        edge_nets = nn.ModuleList()
        sizes = [self.node_embed_dim * 2] + self.edge_hidden_size + [self.edge_embed_dim]
        # print('adj=',self.adj)
        for i in range(self.n_agent):
            for j in range(i + 1, self.n_agent):
                if self.adj[i][j]:
                    edge_nets.append(GraphConvolutionalModel.EdgeNetwork(i, j, sizes))
        return edge_nets

    def _init_node_embedding(self):
        node_embedding = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.state_dim, self.node_embed_dim, output_activation=nn.ReLU)
        state_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, self.state_dim, output_activation=nn.Identity)
        reward_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, 1, nn.Identity)
        done_head = GraphConvolutionalModel.NodeWiseEmbedding(self.n_agent, self.node_embed_dim, 1, nn.Sigmoid)
        return node_embedding, state_head, reward_head, done_head


class QCritic(nn.Module):
    """
    Dueling Q, currently only implemented for discrete action space
    if n_embedding > 0, assumes the action space needs embedding
    Notice that the output shape should be 1+action_space.n for discrete dueling Q
    
    n_embedding is the number of embedding modules needed, = the number of discrete action spaces used as input
    only used for decentralized multiagent, assumes the first action is local (consistent with gather() in utils)
    """

    def __init__(self, env, n_embedding=0, **q_args):
        super().__init__()
        q_net = q_args['network']
        self.action_space = env.action_space
        self.q = q_net(**q_args)
        self.n_embedding = n_embedding
        input_dim = q_args['sizes'][0]
        self.state_per_agent = input_dim // (n_embedding + 1)
        if n_embedding != 0:
            self.action_embedding = nn.Embedding(self.action_space.n, self.state_per_agent)

    def forward(self, state, output_distribution, action=None):
        """
        action is only used for decentralized multiagent
        """
        if isinstance(self.action_space, Box):
            q = self.q(torch.cat([state, action], dim=-1))
        else:
            if self.n_embedding > 0:
                # multiagent
                batch_size, _ = action.shape
                action_embedding = self.action_embedding(action).view(batch_size, -1)
                action_embedding[:, :self.state_per_agent] = 0
                state = state + action_embedding
                action = action[:, 0]
            q = self.q(state)
            while len(q.shape) > 2:
                q = q.squeeze(-1)  # HW of size 1 if CNN
            # [b, a+1]
            v = q[:, -1:]
            q = q[:, :-1]
            # q = q - q.mean(dim=1, keepdim=True) + v
            if output_distribution:
                # q for all actions
                return q
            else:
                # q for a particular action
                q = torch.gather(input=q, dim=1, index=action.unsqueeze(-1))
                return q.squeeze(dim=1)


class QCritic_New(nn.Module):
    def __init__(self, env, n_embedding=0, **q_args):
        super().__init__()
        q_net = q_args['network']
        self.action_space = env.action_space
        self.q = q_net(**q_args)
        self.n_embedding = n_embedding
        input_dim = q_args['sizes'][0]
        #self.state_per_agent = input_dim // (n_embedding + 1)
        if n_embedding != 0:
            #self.action_embedding = nn.Embedding(self.action_space.n, self.state_per_agent)
            self.action_embedding = nn.Embedding(self.action_space.n, self.n_embedding)

    def forward(self, state, output_distribution, action=None):
        """
        action is only used for decentralized multiagent
        """
        if isinstance(self.action_space, Box):
            q = self.q(torch.cat([state, action], dim=-1))
        else:
            if self.n_embedding > 0:
                # multiagent
                batch_size, _ = action.shape
                action_embedding = self.action_embedding(action).view(batch_size, -1)
                #action_embedding[:, :self.state_per_agent] = 0
                state = torch.cat([state, action_embedding], dim=-1)
                #action = action[:, 0]
            q = self.q(state)
            while len(q.shape) > 2:
                q = q.squeeze(-1)  # HW of size 1 if CNN
            return q
            # [b, a+1]
            #v = q[:, -1:]
            #q = q[:, :-1]
            # q = q - q.mean(dim=1, keepdim=True) + v
            #if output_distribution:
                # q for all actions
                #return q
            #else:
                # q for a particular action
                #q = torch.gather(input=q, dim=1, index=action.unsqueeze(-1))
                #return q.squeeze(dim=1)

class CategoricalActor(nn.Module):
    """ 
    always returns a distribution
    """

    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
        self.eps = 1e-5
        # if pi becomes truely deterministic (e.g. SAC alpha = 0)
        # q will become NaN, use eps to increase stability 
        # and make SAC compatible with "Hard"ActorCritic

    def forward(self, obs):
        logit = self.network(obs)
        probs = self.softmax(logit)
        probs = (probs + self.eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs

class CategoricalActor_for_heterogeneous_space(nn.Module):
    """ 
    always returns a distribution
    """

    def __init__(self, heterogeneous_state_dim, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        net_fn = net_args['network']
        self.network = net_fn(heterogeneous_state_dim, **net_args)
        self.eps = 1e-5
        # if pi becomes truely deterministic (e.g. SAC alpha = 0)
        # q will become NaN, use eps to increase stability 
        # and make SAC compatible with "Hard"ActorCritic

    def forward(self, obs):
        logit = self.network(obs)
        probs = self.softmax(logit)
        probs = (probs + self.eps)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return probs


class RegressionActor(nn.Module):
    """
    determinsitc actor, used in DDPG and TD3
    """

    def __init__(self, action_dim, low, high, squeeze=False, **net_args):
        super().__init__()
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
        self.low = low
        self.high = high
        self.output_size = net_args['sizes'][-1]
        self.action_head = nn.Linear(self.output_size, action_dim)
        self.std_head = nn.Linear(self.output_size, action_dim)
        self.squeeze = squeeze
        self.squash = torch

    def forward(self, obs, a=None):
        action, std = self.network(obs)
        std = std.abs()
        distri = Normal(action, std)
        if a is None: # acting
            return

class GaussianActor(nn.Module):
    def __init__(self, action_dim, **net_args):
        super(GaussianActor, self).__init__()
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
        self.output_size = net_args['sizes'][-1]
        self.action_head = nn.Linear(self.output_size, action_dim)
        self.log_std = torch.nn.Parameter(- 0.5 * torch.ones(action_dim, dtype=torch.float32))

    def forward(self, obs, a=None):
        output = self.network(obs)
        mean = self.action_head(output)
        std = torch.exp(self.log_std).expand(*mean.shape)
        if a is None: # acting
            return mean, std
        else:
            distri = Normal(mean, std)
            return distri.log_prob(a).sum(dim=-1)

class SquashedGaussianActor(nn.Module):
    """
    Squashed Gaussian actor used in SAC.
    """

    def __init__(self, action_dim, low, high, squeeze=False, **net_args):
        super(SquashedGaussianActor, self).__init__()
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
        self.low = low
        self.high = high
        self.output_size = net_args['sizes'][-1]
        self.action_head = nn.Linear(self.output_size, action_dim)
        self.std_head = nn.Linear(self.output_size, action_dim)
        self.squeeze = squeeze
        self.squash = net_args['squash']

    def forward(self, obs, a=None):
        output = self.network(obs)
        mean = self.action_head(output)
        std = self.std_head(output)
        std = std.abs()
        distri = Normal(mean, std)
        if a is None: # acting
            a = distri.sample()
            log_p = distri.log_prob(a)
            if self.squash:
                log_p += (log((self.high - self.low) * 2) - a - 2 * torch.log(F.softplus(-2 * a))).sum(dim=-1, keepdim=True)
                a = (self.high + self.low) * 0.5 + (self.high - self.low) * 0.5 * torch.tanh(a)
            else:
                a = torch.clamp(a, self.low, self.high)
            if self.squeeze:
                a = a.squeeze(-1)
            return a, log_p
        else:
            if self.squash:
                original_a = a - (self.high + self.low) * 0.5
                original_a = original_a / (self.high - self.low) * 2
                original_a = 0.5 * (torch.log(1 + original_a) - torch.log(1 - original_a))
                delta_log_p = log((self.high - self.low) * 2) - original_a - 2 * torch.log(F.softplus(-2 * original_a))
                if self.squeeze:
                    delta_log_p = delta_log_p.sum(dim=-1, keepdim=True)
                log_p = distri.log_prob(original_a) + delta_log_p
            else:
                log_p = distri.log_prob(a)
            return log_p

def NLLLoss(state:torch.Tensor, target:torch.Tensor, variance:torch.Tensor, eps=1e-5, reduction='mean'):
    assert state.size() == target.size()
    assert state.size() == variance.size()
    eps = torch.ones_like(variance) * eps
    loss = 0.5 * (torch.max(variance, eps).log() + ((state - target) ** 2) / torch.max(variance, eps))
    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss.mean()


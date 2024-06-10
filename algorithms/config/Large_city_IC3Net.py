from gym.spaces import Box
from numpy import pi
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config
import numpy as np
from gym.spaces import Box, Discrete
import numpy as np
from copy import deepcopy as dp 
def cal_n_order_matrix(n_nodes,max_order,adj):
    def calculate_high_order_adj(max_order):
        result_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
        for i in range(n_nodes):
            for j in range(n_nodes):                  
                if abs(j-i) <= max_order:
                    result_matrix[i][j] = 1 

        return result_matrix

    adjacency_matrix = np.eye(n_nodes)
    result = calculate_high_order_adj(max_order)
    for q in range(n_nodes):
        for k in range(n_nodes):
            if adj[q][k]==1:
                result[q][k]=1
    return result - np.eye(n_nodes)
def getArgs(radius_p, radius_v, radius_pi, env):

    alg_args = Config()
    alg_args.n_iter = 25000
    alg_args.n_inner_iter = 1
    alg_args.n_warmup = 0
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 1
    alg_args.model_validate_interval = 10
    alg_args.test_interval = 10
    alg_args.rollout_length = 500
    alg_args.test_length = 500
    alg_args.max_episode_len = 500
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = 'checkpoints/standard_makeFigureEight2_MB_DPPOAgent_17361/81501_5222.7847817614875.pt'
    alg_args.n_traj = 2048
    alg_args.model_traj_length = 8
    alg_args.model_error_thres = 0.
    alg_args.model_batch_size = 256
    alg_args.model_buffer_size = 15
    alg_args.model_update_length = 2
    alg_args.model_length_schedule = lambda x: min(25, 8 + int(x/4))

    agent_args = Config()
    adj_order = 20
    agent_args.adj = cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
    agent_args.n_agent = agent_args.adj.shape[0]
    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 7.5e-3
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.05
    agent_args.entropy_coeff_decay = agent_args.entropy_coeff / 5e5  # only in IA2C
    agent_args.lr = 5e-4
    agent_args.lr_v = 5e-4
    agent_args.lr_p = 1e-2  # since update time is lower
    agent_args.n_update_v = 30
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = False  # just use advantage rather than the reduced
    agent_args.use_rtg = False
    agent_args.use_gae_returns = False # set to false
    agent_args.advantage_norm = True
    agent_args.hidden_state_dim = 8
    agent_args.embedding_sizes = [env.n_s, 16, agent_args.hidden_state_dim]
    agent_args.observation_dim = env.n_s
    agent_args.action_space = env.action_space
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p
    agent_args.squeeze = True

    p_args = Config()
    p_args.n_conv = 1
    p_args.n_embedding = 0
    p_args.residual = True
    p_args.edge_embed_dim = 12
    p_args.node_embed_dim = 8
    p_args.edge_hidden_size = [16, 16]
    p_args.node_hidden_size = [16, 16]
    p_args.reward_coeff = 10.0
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 128, 128, 1]
    v_args.hidden_dim = 64
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 128, 128, env.n_action]
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args
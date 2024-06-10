from gym.spaces import Box
from numpy import pi
import torch.nn
from algorithms.models import MLP
from algorithms.models import MLP_for_heterogeneous_space
from algorithms.utils import Config
import numpy as np
from copy import deepcopy as dp 
# def get_n_order_adj(n_agent, order, adj_matrix, new_adj_matrix):
    
#     def adj_order(adj_matrix, new_adj_matrix):
#         for i in range(n_agent):
#             for j in range(n_agent):
#                 if adj_matrix[i][j] == 1:
#                     a = [index for index, value in enumerate(adj_matrix[j]) if value != 0]
#                     for q in range(len(a)):
#                         new_adj_matrix[i][a[q]] = 1
#         return new_adj_matrix
    
    
#     def cal_n_order_matrix(order, adj_matrix, new_adj_matrix):
    
#         for i in range(order-1):
#             a = adj_order(adj_matrix, new_adj_matrix)
#             adj_matrix = dp(a)
#             new_adj_matrix = dp(a)
            
#             if i == order-2:
#                 b = a - np.eye(n_agent)
#                 return b 
            
        
#     b = cal_n_order_matrix(order, adj_matrix, new_adj_matrix)
#     return b

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
    alg_args.test_interval = 10
    alg_args.rollout_length = 500     # 500 best
    alg_args.test_length = alg_args.rollout_length
    alg_args.max_episode_len = alg_args.rollout_length
    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 128
    alg_args.model_buffer_size = 0

    agent_args = Config()
    adj_order = 10
    #print("lennnnnnn = ",sum(env.neighbor_mask[0]))
    agent_args.adj = cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
    agent_args.n_agent = agent_args.adj.shape[0]
    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.0
    agent_args.lr = 5e-4
    agent_args.lr_v = 5e-4
    agent_args.n_update_v = 30
    agent_args.n_update_pi = 10
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.use_rtg = True
    agent_args.use_gae_returns = False   #False
    agent_args.advantage_norm = True
    #agent_args.observation_space = env.observation_space
    agent_args.observation_dim = env.n_s
    agent_args.action_space = env.action_space
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p
    agent_args.squeeze = False

    p_args = None
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 128, 128, 1]
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 128, 128, env.n_action]
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args

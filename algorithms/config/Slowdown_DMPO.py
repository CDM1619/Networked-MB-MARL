from gym.spaces import Box
from numpy import pi
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config


def getArgs(radius_p, radius_v, radius_pi, env):

    alg_args = Config()
    alg_args.n_iter = 25000
    alg_args.n_inner_iter = 8
    alg_args.n_warmup = 50
    alg_args.n_model_update = int(5e4)
    alg_args.n_model_update_warmup = int(5e4)

    #alg_args.n_warmup = 1
    #alg_args.n_model_update = int(5)
    #alg_args.n_model_update_warmup = int(2e1)

    alg_args.n_test = 5
    alg_args.model_validate_interval = 3
    alg_args.test_interval = 20
    alg_args.rollout_length = 600
    alg_args.test_length = 600
    alg_args.max_episode_len = 600
    alg_args.model_based = True
    alg_args.load_pretrained_model = False
    
    alg_args.pretrained_model = 'D:/A_RL/MB-MARL/checkpoints/standard _CACC_slowdown_MB_DPPOAgent_23243/2572680_-2813.359473800659.pt'

    alg_args.n_traj = 1024
    alg_args.model_traj_length = 25
    alg_args.model_error_thres = 2e-5
    alg_args.model_prob = 0.5
    alg_args.model_batch_size = 512
    alg_args.model_buffer_size = 15
    alg_args.model_update_length = 4
    alg_args.model_length_schedule = None

    agent_args = Config()
    agent_args.adj = env.neighbor_mask
    agent_args.n_agent = agent_args.adj.shape[0]

    #import numpy as np
    #agent_args.adj = np.eye(agent_args.n_agent)


    #import numpy as np
    #agent_args.adj  = np.eye(agent_args.n_agent)
    #for i in range(agent_args.n_agent):
        #for j in range(agent_args.n_agent):
            #if abs(j-i)==agent_args.n_agent//2:
                #agent_args.adj[i][j]=1


    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 7.5e-2
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 1.0
    agent_args.lr = 5e-4
    agent_args.lr_v = 5e-4
    agent_args.lr_p = 5e-4
    agent_args.n_update_v = 15 # deprecated
    agent_args.n_update_pi = 8
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.use_rtg = False
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True
    agent_args.observation_space = env.observation_space
    agent_args.hidden_state_dim = 8
    agent_args.embedding_sizes = [env.observation_space.shape[0], 16, agent_args.hidden_state_dim]
    agent_args.observation_dim = 5
    agent_args.action_space = env.action_space
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p
    agent_args.squeeze = True

    p_args = Config()
    p_args.n_conv = 1
    p_args.n_embedding = 4
    p_args.residual = True
    p_args.edge_embed_dim = 12
    p_args.node_embed_dim = 8
    p_args.edge_hidden_size = [128, 128]
    p_args.node_hidden_size = [128, 128]
    p_args.reward_coeff = 10.
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 64, 64, 1]
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 64, 64, agent_args.action_space.n]
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args

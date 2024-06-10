from gym.spaces import Box
from numpy import pi
import torch.nn
from algorithms.models import MLP
from algorithms.utils import Config
from gym.spaces import Box
import numpy as np



def getArgs(radius_p, radius_v, radius_pi, env):

    alg_args = Config()
    alg_args.n_iter = 25000
    alg_args.n_inner_iter = 10
    alg_args.n_warmup = 10                ##  50
    #alg_args.n_model_update = int(500)
    alg_args.n_model_update = int(2e3)
    alg_args.n_model_update_warmup = int(2e4)


    #alg_args.n_warmup = 1
    #alg_args.n_model_update = int(1)
    #alg_args.n_model_update_warmup = int(1)

    alg_args.n_test = 5
    alg_args.model_validate_interval = 10
    alg_args.test_interval = 20


    alg_args.rollout_length = 240
    alg_args.test_length = 240
    alg_args.max_episode_len = 240

    alg_args.model_based = True
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = 'checkpoints/standard _makeRingAttenuation_MB_DPPOAgent_17293/2054577_-551.1475067236545.pt'
    # alg_args.pretrained_model = 'checkpoints/standard _makeRingAttenuation_MB_DPPOAgent_5715/165000_-5999.50454407513.pt'
    alg_args.n_traj = 2048
    alg_args.model_traj_length = 15     
    alg_args.model_error_thres = 2e-10
    alg_args.model_prob = 0.5
    alg_args.model_batch_size = 256           #最佳256
    alg_args.model_buffer_size = 15
    alg_args.model_update_length = 4            #最佳4
    alg_args.model_length_schedule = None

    agent_args = Config()



    #def init_neighbor_mask():
        #n = env.n_agents
        #neighbor_mask = np.zeros((n, n))
        #for i in range(n):
            #neighbor_mask[i][i] = 1
            #neighbor_mask[i][(i+1)%n] = 1
            #neighbor_mask[i][(i+n-1)%n] = 1
        #return neighbor_mask
    #agent_args.adj = init_neighbor_mask()


    agent_args.adj = np.eye(env.n_agents)
    agent_args.n_agent = env.n_agents

    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.0
    agent_args.lr = 5e-4
    agent_args.lr_v = 5e-4
    #agent_args.lr = 1e-5
    #agent_args.lr_v = 1e-5
    
    agent_args.lr_p = 5e-5
    agent_args.n_update_v = 5
    agent_args.n_update_pi = 5
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.use_rtg = True               #最佳False
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True
    #agent_args.observation_space = env.observation_space
    agent_args.hidden_state_dim = 128          #最佳64
    agent_args.embedding_sizes = [env.obs_size, 128, agent_args.hidden_state_dim]        #最佳64
    agent_args.observation_dim = env.obs_size
    
    agent_args.action_space = Box(low=env.action_space.low, high=env.action_space.high, shape=(np.max(np.count_nonzero(env.action_mask, axis=1)), ), dtype=np.float32)
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p
    agent_args.squeeze = True

    p_args = Config()
    p_args.n_conv = 1
    p_args.n_embedding = 0
    p_args.residual = True
    p_args.edge_embed_dim = 128              #  最佳48
    p_args.node_embed_dim = 128              #   最佳32
    p_args.edge_hidden_size = [128, 128]    #128  128 for 最简单
    p_args.node_hidden_size = [128, 128]
    p_args.reward_coeff = 10.0
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 512, 512, 1]           #512  512 for 最简单
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 512, 512, 16]              #512  512 for 最简单
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args












   # for 199 agents

    p_args = Config()
    p_args.n_conv = 1
    p_args.n_embedding = 0
    p_args.residual = True
    p_args.edge_embed_dim = 64              #  最佳48
    p_args.node_embed_dim = 64              #   最佳32
    p_args.edge_hidden_size = [128, 128]    #128  128 for 最简单
    p_args.node_hidden_size = [128, 128]
    p_args.reward_coeff = 10.0
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 256, 256, 1]           #512  512 for 最简单
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 256, 256, 16]              #512  512 for 最简单
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args














    return alg_args
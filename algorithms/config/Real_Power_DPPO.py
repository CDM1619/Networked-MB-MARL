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
    alg_args.n_inner_iter = 1
    alg_args.n_warmup = 0
    alg_args.n_model_update = 5
    alg_args.n_model_update_warmup = 10
    alg_args.n_test = 5
    alg_args.test_interval = 20


    alg_args.rollout_length = 240
    alg_args.test_length = 240
    alg_args.max_episode_len = 240

    alg_args.model_based = False
    alg_args.load_pretrained_model = False
    alg_args.pretrained_model = None
    alg_args.model_batch_size = 256
    alg_args.model_buffer_size = 0

    agent_args = Config()

    #def init_neighbor_mask(n):
        #neighbor_mask = np.zeros((n, n))
        #for i in range(n):
            #neighbor_mask[i][i] = 1
            #neighbor_mask[i][(i+1)%n] = 1
            #neighbor_mask[i][(i+n-1)%n] = 1
        #return neighbor_mask
    
        

    #agent_args.adj = env.neighbor_mask
    agent_args.adj = np.eye(env.n_agents)
    agent_args.n_agent = env.n_agents
    agent_args.gamma = 0.99
    agent_args.lamda = 0.5
    agent_args.clip = 0.2
    agent_args.target_kl = 0.01
    agent_args.v_coeff = 1.0
    agent_args.v_thres = 0.
    agent_args.entropy_coeff = 0.0
    #agent_args.lr = 1e-3
    #agent_args.lr_v = 1e-3

    agent_args.lr = 5e-4
    agent_args.lr_v = 5e-4

    agent_args.n_update_v = 5
    agent_args.n_update_pi = 5
    agent_args.n_minibatch = 1
    agent_args.use_reduced_v = True
    agent_args.use_rtg = True
    agent_args.use_gae_returns = False
    agent_args.advantage_norm = True
    #agent_args.observation_space = env.observation_space
    agent_args.observation_dim = env.obs_size
    #agent_args.action_space = env.action_space
    

    agent_args.action_space = Box(low=env.action_space.low, high=env.action_space.high, shape=(np.max(np.count_nonzero(env.action_mask, axis=1)), ), dtype=np.float32)
    
    #agent_args.action_space = Box(low=env.action_space.low, high=env.action_space.high, shape=(6, ), dtype=np.float32)



            
    agent_args.radius_v = radius_v
    agent_args.radius_pi = radius_pi
    agent_args.radius_p = radius_p
    agent_args.squeeze = True

    p_args = None
    agent_args.p_args = p_args

    v_args = Config()
    v_args.network = MLP
    v_args.activation = torch.nn.ReLU
    v_args.sizes = [-1, 512, 512, 1]      ##  256  256
    agent_args.v_args = v_args

    pi_args = Config()
    pi_args.network = MLP
    pi_args.activation = torch.nn.ReLU
    pi_args.sizes = [-1, 512, 512, 16]     ##  256  256
    pi_args.squash = False
    agent_args.pi_args = pi_args

    alg_args.agent_args = agent_args

    return alg_args

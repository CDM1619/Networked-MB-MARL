import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from logging import log
import os
from re import T
import importlib
#import ray
import time
import warnings
import json
import gym

os.environ["WANDB_API_KEY"] = 'your wandb key'

os.environ["WANDB_TEAM"] = ""



from algorithms.utils import Config, LogClient, LogServer, mem_report
from algorithms.envs.FigureEight import makeFigureEight2, makeFigureEightTest
from algorithms.envs.CACC import CACC_catchup, CACC_slowdown, CACC_catchup_test, CACC_slowdown_test
from algorithms.envs.ATSC import Grid_Env
from algorithms.envs.ATSC import Monaco_Env
from algorithms.envs.Large_net import Large_city_Env
from torch import distributed as dist



from algorithms.algo.main import OnPolicyRunner
os.environ['MKL_SERVICE_FORCE_INTEL']='1'
import torch
import argparse




warnings.filterwarnings('ignore')

def getEnvArgs():
    env_args = Config()
    env_args.n_env = 1
    env_args.n_cpu = 1 # per environment
    env_args.n_gpu = 0
    return env_args

def getRunArgs(input_args):
    run_args = Config()
    run_args.n_thread = 1
    run_args.parallel = False
    
    run_args.device = input_args.device
 
    run_args.n_cpu = 1/4
    run_args.n_gpu = 0
    run_args.debug = False
    run_args.test = False
    run_args.profiling = False
    run_args.name = f'standard{input_args.name}'
    
    
    
    run_args.radius_v = 1
    run_args.radius_pi = 1
    run_args.radius_p = 1 
    
    

    
    run_args.init_checkpoint = None
    run_args.start_step = 0
    run_args.save_period = 1800 # in seconds
    run_args.log_period = int(20)
    run_args.seed = None
    return run_args

def initArgs(run_args, env_train, env_test, input_arg):
    ref_env = env_train

    if input_arg.env in ['eight', 'ring', 'catchup', 'slowdown','Grid','Monaco','PowerGrid', "Real_Power", "Pandemic","Large_city"] or input_arg.algo in ['CPPO', 'DMPO', 'IC3Net', 'IA2C']:
        env_str = input_arg.env[0].upper() + input_arg.env[1:]
        config = importlib.import_module(f"algorithms.config.{env_str}_{input_args.algo}")

    if input_arg.env in ['catchup', 'slowdown']:
        run_args.radius_v = 4
        run_args.radius_pi = 1
        run_args.radius_p = 1


    if input_arg.env in ['Monaco']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1
        
    if input_arg.env in ['Grid']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1



    if input_arg.env in ['PowerGrid']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1

    if input_arg.env in ['Real_Power']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1

    if input_arg.env in ['Pandemic']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1

    if input_arg.env in ['Large_city']:
        run_args.radius_v = 1
        run_args.radius_pi = 1
        run_args.radius_p = 1      

    if input_arg.algo in ['CPPO']:
        run_args.radius_v = env_train.n_agent # n_agent
        run_args.radius_pi = 1
        run_args.radius_p = 1
        
  
        


    alg_args = config.getArgs(run_args.radius_p, run_args.radius_v, run_args.radius_pi, ref_env)
    return alg_args

def initAgent(logger, device, agent_args, input_args):
    return agent_fn(logger, device, agent_args, input_args)

def initEnv(input_args):
    if input_args.env == 'eight':
        env_fn_train, env_fn_test = makeFigureEight2, makeFigureEightTest
    elif input_args.env == 'ring':
        from algorithms.envs.Ring2 import makeRingAttenuation, makeRingAttenuationTest
        env_fn_train, env_fn_test = makeRingAttenuation, makeRingAttenuationTest 
    elif input_args.env == 'catchup':
        env_fn_train, env_fn_test = CACC_catchup, CACC_catchup_test            
    elif input_args.env == 'slowdown':
        env_fn_train, env_fn_test = CACC_slowdown, CACC_slowdown_test
    elif input_args.env == 'Grid':
        env_fn_train, env_fn_test = Grid_Env, Grid_Env  
    elif input_args.env == 'Monaco':
        env_fn_train, env_fn_test = Monaco_Env, Monaco_Env  
    elif input_args.env == 'PowerGrid':
        from algorithms.envs.PowerGrid_ENV import PowerGrid_Env
        env_fn_train, env_fn_test = PowerGrid_Env, PowerGrid_Env
    elif input_args.env == 'Real_Power':
        from algorithms.envs.Real_Power import Real_Power_Env
        env_fn_train, env_fn_test = Real_Power_Env, Real_Power_Env
    elif input_args.env == 'Pandemic':
        from algorithms.envs.Pandemic_ENV import Pandemic
        env_fn_train, env_fn_test = Pandemic, Pandemic
    elif input_args.env == 'Large_city':
        env_fn_train, env_fn_test = Large_city_Env, Large_city_Env 
    



    else:
        env_fn_train, env_fn_test = None
    return env_fn_train, env_fn_test

def override(alg_args, run_args, env_fn_train, input_args):
    alg_args.env_fn = env_fn_train
    agent_args = alg_args.agent_args
    p_args, v_args, pi_args = agent_args.p_args, agent_args.v_args, agent_args.pi_args
    if run_args.debug:
        alg_args.model_batch_size = 4
        alg_args.max_ep_len=5
        alg_args.rollout_length = 5
        alg_args.test_length = 1
        alg_args.model_buffer_size = 10
        alg_args.n_model_update = 3
        alg_args.n_model_update_warmup = 3
        alg_args.n_warmup = 1
        alg_args.n_test = 1
        alg_args.n_traj = 4
        alg_args.n_inner_iter = 10
    if run_args.test:
        alg_args.n_warmup = 0
        alg_args.n_test = 10
    if run_args.profiling:
        alg_args.model_batch_size = 128
        alg_args.n_warmup = 0
        if alg_args.agent_args.p_args is None:
            alg_args.n_iter = 10
        else:
            alg_args.n_iter = 10
            alg_args.model_buffer_size = 1000
            alg_args.n_warmup = 1
        alg_args.n_test = 1
        alg_args.max_ep_len = 400
        alg_args.rollout_length = 400
        alg_args.test_length = 1
        alg_args.test_interval = 100
    if run_args.seed is None:
        run_args.seed = int(time.time()*1000)%65536
    agent_args.parallel = run_args.parallel
    agent_args.lable_name=input_args.algo+input_args.name
    ## update the parameter from the input arg
    for key in input_args.para:
        key_ls = key.split('.')
        *pre_key_ls, key_last = key_ls
        target_args = alg_args
        for pre_key in pre_key_ls:
            target_args = target_args.__dict__[pre_key]
        target_args.__dict__[key_last] = input_args.para[key]
    #run_args.name = '{}_{}_{}_{}'.format(run_args.name, env_fn_train.__name__, agent_fn.__name__, run_args.seed)
    run_args.name = '{}_{}'.format(agent_fn.__name__, run_args.seed)
    return alg_args, run_args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, required=False, default='catchup', help="environment(eight/ring/catchup/slowdown/Grid/Monaco/PowerGrid/Real_Power/Pandemic/Large_city)")
    parser.add_argument('--algo', type=str, required=False, default='DMPO', help="algorithm(DMPO/IC3Net/CPPO/DPPO/IA2C) ")
    parser.add_argument('--device', type=str, required=False, default='cuda:6', help="device(cpu/cuda:0/cuda:1/...) ")
    parser.add_argument('--name', type=str, required=False, default='', help="the additional name for logger")
    parser.add_argument('--para', type=str, required=False, default='{}', help="the hyperparameter json string" )
    
    args = parser.parse_args()
    args.para = json.loads(args.para.replace('\'', '\"'))
    '''
    if not args.option:
        parser.print_help()
        exit(1)
    '''
    return args


# get arg from cli
input_args = parse_args()

# import agent [must put here, if in a function, import will become local]
if input_args.algo == 'IA2C':
    from algorithms.algo.agent.IA2C import IA2C as agent_fn
elif input_args.algo == 'IC3Net':
    from algorithms.algo.agent.IC3Net import IC3Net as agent_fn
elif input_args.algo == 'DPPO':
    from algorithms.algo.agent.DPPO import DPPOAgent as agent_fn
elif input_args.algo == 'CPPO':
    from algorithms.algo.agent.CPPO import CPPOAgent as agent_fn
elif input_args.algo == 'DMPO':
    from algorithms.algo.agent.DMPO import DMPOAgent as agent_fn

env_args = getEnvArgs()
env_fn_train, env_fn_test = initEnv(input_args)


env_train = env_fn_train()
env_test = env_train


######################################33333

#import configparser
#config_dir = './deeprl_network/config/config_ia2c_catchup.ini'

#config = configparser.ConfigParser()
#config.read(config_dir)

### init env
#import sys
#sys.path.append("/home/chengdong")
#from deeprl_network.envs.cacc_env import CACCEnv
#env_train = CACCEnv(config['ENV_CONFIG'])
#env_test = CACCEnv(config['ENV_CONFIG'])
######################################33333  



run_args = getRunArgs(input_args)
alg_args = initArgs(run_args, env_train, env_test, input_args)
alg_args, run_args = override(alg_args, run_args, env_fn_train, input_args)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


logger = LogServer({'run_args':run_args, 'algo_args':alg_args}, mute=run_args.debug or run_args.test or run_args.profiling)
logger = LogClient(logger)
agent = initAgent(logger, run_args.device, alg_args.agent_args, input_args)

# torch.set_num_threads(run_args.n_thread)
print(f"n_threads {torch.get_num_threads()}")
print(f"n_gpus {torch.cuda.device_count()}")

if run_args.profiling:
    import cProfile
    cProfile.run("OnPolicyRunner(logger = logger, run_args=run_args, alg_args=alg_args, agent=agent, env_learn=env_train, env_test = env_test).run()",
                  filename=f'device{run_args.device}_parallel{run_args.parallel}.profile')
else:
    OnPolicyRunner(logger = logger, run_args=run_args, alg_args=alg_args, agent=agent, env_learn=env_train, env_test = env_test,env_args=input_args).run()



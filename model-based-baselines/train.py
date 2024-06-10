import argparse
import os
import torch
import socket
import setproctitle
import wandb

from agent.runners.DreamerRunner import DreamerRunner
from configs import Experiment #, SimpleObservationConfig, NearRewardConfig, DeadlockPunishmentConfig, RewardsComposerConfig
#from configs.EnvConfigs import StarCraftConfig, EnvCurriculumConfig, CACCConfig, ATSCConfig
from configs.EnvConfigs import EnvCurriculumConfig, CACCConfig, ATSCConfig, PowerGridConfig
# from configs.flatland.RewardConfigs import FinishRewardConfig
from configs.dreamer.DreamerControllerConfig import DreamerControllerConfig
from configs.dreamer.DreamerLearnerConfig import DreamerLearnerConfig
# from configs.flatland.TimetableConfigs import AllAgentLauncherConfig
# from env.flatland.params import SeveralAgents, PackOfAgents, LotsOfAgents
from environments import Env#, FlatlandType, FLATLAND_OBS_SIZE, FLATLAND_ACTION_SIZE

import sys
import time
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from algorithms.envs.CACC import CACC_catchup_3, CACC_slowdown_3, CACC_catchup_test, CACC_slowdown_test
from algorithms.envs.ATSC import Grid_Env_3, Monaco_Env_3
from algorithms.envs.PowerGrid.envs.Grid_envs import GridEnv_3
from algorithms.envs.Pandemic_ENV import Pandemic
from algorithms.utils import Config, LogClient, LogServer, mem_report
import os
os.environ["WANDB_API_KEY"] = '4c1c4bb4621d16d5d7dc6ab8cba6687c54cb8638'
os.environ["WANDB_TEAM"] = ""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="atsc", help='cacc / atsc / powergrid')
    parser.add_argument('--env_name', type=str, default="Grid", help='slowdown / catchup / Grid / Monaco')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers')
    return parser.parse_args()


def train_dreamer(logger, env, exp, n_workers):
    runner = DreamerRunner(logger, env, exp.env_config, exp.learner_config, exp.controller_config, n_workers)
    runner.run(exp.steps, exp.episodes)


def get_env_info(configs, env):
    for config in configs:
        config.IN_DIM = env.n_obs
        config.ACTION_SIZE = env.n_actions
        config.n_ags = env.n_agents
    env.close()


def get_env_info_flatland(configs):
    for config in configs:
        config.IN_DIM = FLATLAND_OBS_SIZE
        config.ACTION_SIZE = FLATLAND_ACTION_SIZE


def prepare_starcraft_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = StarCraftConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_cacc_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = CACCConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_atsc_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = ATSCConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_powergrid_configs(env_name):
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    env_config = PowerGridConfig(env_name)
    get_env_info(agent_configs, env_config.create_env())
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": None,
            "obs_builder_config": None}

def prepare_flatland_configs(env_name):
    if env_name == FlatlandType.FIVE_AGENTS:
        env_config = SeveralAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.TEN_AGENTS:
        env_config = PackOfAgents(RANDOM_SEED + 100)
    elif env_name == FlatlandType.FIFTEEN_AGENTS:
        env_config = LotsOfAgents(RANDOM_SEED + 100)
    else:
        raise Exception("Unknown flatland environment")
    obs_builder_config = SimpleObservationConfig(max_depth=3, neighbours_depth=3,
                                                 timetable_config=AllAgentLauncherConfig())
    reward_config = RewardsComposerConfig((FinishRewardConfig(finish_value=10),
                                           NearRewardConfig(coeff=0.01),
                                           DeadlockPunishmentConfig(value=-5)))
    agent_configs = [DreamerControllerConfig(), DreamerLearnerConfig()]
    get_env_info_flatland(agent_configs)
    return {"env_config": (env_config, 100),
            "controller_config": agent_configs[0],
            "learner_config": agent_configs[1],
            "reward_config": reward_config,
            "obs_builder_config": obs_builder_config}


if __name__ == "__main__":
    RANDOM_SEED = torch.randint(0, 10000, (1,)).item()
    args = parse_args()
    #args.env = 'cacc'
    #args.env_name = 'slowdown'
    args.cuda_num = '5'
    torch.set_num_threads(10)
    #os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_num
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    if args.env == Env.FLATLAND:
        configs = prepare_flatland_configs(args.env_name)
    elif args.env == Env.STARCRAFT:
        configs = prepare_starcraft_configs(args.env_name)
    elif args.env == Env.CACC:
        configs = prepare_cacc_configs(args.env_name)
    elif args.env == Env.ATSC:
        configs = prepare_atsc_configs(args.env_name)
    elif args.env == Env.PowerGrid:
        configs = prepare_powergrid_configs(args.env_name)
    else:
        raise Exception("Unknown environment")
    configs["env_config"][0].ENV_TYPE = Env(args.env)
    configs["learner_config"].ENV_TYPE = Env(args.env)
    configs["controller_config"].ENV_TYPE = Env(args.env)

    if configs["learner_config"].use_wandb:
        wandb.init(config=configs["learner_config"],
                    project='Multi-Agent Ensemble',
                    entity='drl1619',
                    notes=socket.gethostname(),
                    name='S4_' + str(RANDOM_SEED) + '_' + args.cuda_num,
                    group=args.env_name,
                    dir=configs["learner_config"].LOG_FOLDER,
                    job_type="training",
                    reinit=True)
        wandb.define_metric('total_step')
        wandb.define_metric('incre_win_rate', step_metric='total_step')
        wandb.define_metric('aver_step_reward', step_metric='total_step')
        setproctitle.setproctitle(str(RANDOM_SEED) + '_' + args.cuda_num)

    exp = Experiment(steps=int(1e6),
                     episodes=50000,
                     random_seed=RANDOM_SEED,
                     env_config=EnvCurriculumConfig(*zip(configs["env_config"]), Env(args.env),
                                                    obs_builder_config=configs["obs_builder_config"],
                                                    reward_config=configs["reward_config"]),
                     controller_config=configs["controller_config"],
                     learner_config=configs["learner_config"])
                     
                     
    #### wandb ###3###############################################33
    #alg_args = Config()   
    #if args.env_name == "catchup":
        #alg_args.env_fn = CACC_catchup_3 
    #elif args.env_name == "slowdown":
        #alg_args.env_fn = CACC_slowdown_3 
    #elif args.env_name == "Grid":
        #alg_args.env_fn = Grid_Env_3
    #elif args.env_name == "Monaco":
        #alg_args.env_fn = Monaco_Env_3
    #elif args.env_name == "PowerGrid":
        #alg_args.env_fn = GridEnv_3
    
    #alg_args.env_name = args.env_name
    #alg_args.algo_name = "MAG"  
    #run_args = Config()
    #run_args.radius_v = 1
    #run_args.radius_pi = 1
    #run_args.radius_p = 1 
    #run_args.debug = False
    #run_args.name = alg_args.env_name + "_" + alg_args.algo_name
    #run_args.save_period = 1800
    #run_args.log_period = int(20)
    #run_args.seed = int(time.time()*1000)%65536
    #logger = LogServer({'run_args':run_args, 'algo_args':alg_args}, mute=run_args.debug)
    #logger = LogClient(logger)
    #### wandb ###3###############################################33
    logger = None

    train_dreamer(logger, args.env_name, exp, n_workers=args.n_workers)

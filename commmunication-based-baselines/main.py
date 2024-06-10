"""
Main function for training and evaluating MARL algorithms in NMARL envs
@author: Tianshu Chu
"""
import random
import os
import argparse
import configparser
import logging
import threading
from torch.utils.tensorboard.writer import SummaryWriter
from envs.cacc_env import CACCEnv
from envs.large_grid_env import LargeGridEnv
from envs.real_net_env import RealNetEnv
from agents.models import IA2C, IA2C_FP, MA2C_NC, IA2C_CU, MA2C_CNET, MA2C_DIAL
from utils import (Counter, Trainer, Tester, Evaluator,
                   check_dir, copy_file, find_file,
                   init_dir, init_log, init_test_flag)

import sys
import numpy as np
import time
import importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
from algorithms.envs.CACC import CACC_catchup_2, CACC_slowdown_2, CACC_catchup_test, CACC_slowdown_test
from algorithms.envs.ATSC import Grid_Env_2, Monaco_Env_2
from algorithms.envs.PowerGrid.envs.Grid_envs import GridEnv_2
from algorithms.envs.PowerGrid_ENV import PowerGrid_Env_2
from algorithms.envs.Pandemic_ENV import Pandemic_2
from algorithms.envs.Large_net import Large_city_Env


from algorithms.utils import Config, LogClient, LogServer, mem_report
import os
os.environ["WANDB_API_KEY"] = '4c1c4bb4621d16d5d7dc6ab8cba6687c54cb8638'
os.environ["WANDB_TEAM"] = ""

algo_name = "NeurComm"     #   CommNet     DIAL     NeurComm     ConseNet     IA2C  FPrint 
env_name = "PowerGrid"     #  catchup     slowdown     Grid    Monaco   PowerGrid  Pandemic Large_city




#config = importlib.import_module(f"algorithms.config.{env_str}_{input_args.algo}")





def parse_args(env_name, algo_name):


    if env_name == "catchup" and algo_name == "IA2C":
        default_base_dir = current_dir + '/ia2c_catchup_1.0'
        default_config_dir = './config/config_ia2c_catchup.ini'
    elif env_name == "catchup" and algo_name == "FPrint":
        default_base_dir = current_dir + '/ia2c_fp_catchup_1.0'
        default_config_dir = './config/config_ia2c_fp_catchup.ini'
    elif env_name == "catchup" and algo_name == "CommNet":
        default_base_dir = current_dir + '/ma2c_cnet_catchup_1.0'
        default_config_dir = './config/config_ma2c_cnet_catchup.ini'
    elif env_name == "catchup" and algo_name == "DIAL":
        default_base_dir = current_dir + '/ma2c_dial_catchup_1.0'
        default_config_dir = './config/config_ma2c_dial_catchup.ini'
    elif env_name == "catchup" and algo_name == "NeurComm":
        default_base_dir = current_dir + '/ma2c_nc_catchup_1.0'
        default_config_dir = './config/config_ma2c_nc_catchup.ini'
    elif env_name == "catchup" and algo_name == "ConseNet":
        default_base_dir = current_dir + '/ia2c_cu_catchup_1.0'
        default_config_dir = './config/config_ia2c_cu_catchup.ini'

    elif env_name == "slowdown" and algo_name == "IA2C":
        default_base_dir = current_dir + '/ia2c_slowdown_1.0'
        default_config_dir = './config/config_ia2c_slowdown.ini'
    elif env_name == "slowdown" and algo_name == "FPrint":
        default_base_dir = current_dir + '/ia2c_fp_slowdown_1.0'
        default_config_dir = './config/config_ia2c_fp_slowdown.ini'
    elif env_name == "slowdown" and algo_name == "CommNet":
        default_base_dir = current_dir + '/ma2c_cnet_slowdown_1.0'
        default_config_dir = './config/config_ma2c_cnet_slowdown.ini'
    elif env_name == "slowdown" and algo_name == "DIAL":
        default_base_dir = current_dir + '/ma2c_dial_slowdown_1.0'
        default_config_dir = './config/config_ma2c_dial_slowdown.ini'
    elif env_name == "slowdown" and algo_name == "NeurComm":
        default_base_dir = current_dir + '/ma2c_nc_slowdown_1.0'
        default_config_dir = './config/config_ma2c_nc_slowdown.ini'
    elif env_name == "slowdown" and algo_name == "ConseNet":
        default_base_dir = current_dir + '/ia2c_cu_slowdown_1.0'
        default_config_dir = './config/config_ia2c_cu_slowdown.ini'

    elif env_name == "Grid" and algo_name == "IA2C":
        default_base_dir = current_dir + '/ia2c_grid_1.0'
        default_config_dir = './config/config_ia2c_grid.ini'
    elif env_name == "Grid" and algo_name == "FPrint":
        default_base_dir = current_dir + '/ia2c_fp_grid_1.0'
        default_config_dir = './config/config_ia2c_fp_grid.ini'
    elif env_name == "Grid" and algo_name == "CommNet":
        default_base_dir = current_dir + '/ma2c_cnet_grid_1.0'
        default_config_dir = './config/config_ma2c_cnet_grid.ini'
    elif env_name == "Grid" and algo_name == "DIAL":
        default_base_dir = current_dir + '/ma2c_dial_grid_1.0'
        default_config_dir = './config/config_ma2c_dial_grid.ini'
    elif env_name == "Grid" and algo_name == "NeurComm":
        default_base_dir = current_dir + '/ma2c_nc_grid_1.0'
        default_config_dir = './config/config_ma2c_nc_grid.ini'
    elif env_name == "Grid" and algo_name == "ConseNet":
        default_base_dir = current_dir + '/ia2c_cu_grid_1.0'
        default_config_dir = './config/config_ia2c_cu_grid.ini'

    elif env_name == "Monaco" and algo_name == "IA2C":
        default_base_dir = current_dir + '/ia2c_net_1.0'
        default_config_dir = './config/config_ia2c_net.ini'
    elif env_name == "Monaco" and algo_name == "FPrint":
        default_base_dir = current_dir + '/ia2c_fp_net_1.0'
        default_config_dir = './config/config_ia2c_fp_net.ini'
    elif env_name == "Monaco" and algo_name == "CommNet":
        default_base_dir = current_dir + '/ma2c_cnet_net_1.0'
        default_config_dir = './config/config_ma2c_cnet_net.ini'
    elif env_name == "Monaco" and algo_name == "DIAL":
        default_base_dir = current_dir + '/ma2c_dial_net_1.0'
        default_config_dir = './config/config_ma2c_dial_net.ini'
    elif env_name == "Monaco" and algo_name == "NeurComm":
        default_base_dir = current_dir + '/ma2c_nc_net_1.0'
        default_config_dir = './config/config_ma2c_nc_net.ini'
    elif env_name == "Monaco" and algo_name == "ConseNet":
        default_base_dir = current_dir + '/ia2c_cu_net_1.0'
        default_config_dir = './config/config_ia2c_cu_net.ini'




    elif env_name == "PowerGrid" and algo_name == "CommNet":
        default_base_dir = current_dir +'/ma2c_cnet_powergrid_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ma2c_cnet_DER6.ini'
    elif env_name == "PowerGrid" and algo_name == "DIAL":
        default_base_dir = current_dir +'/ma2c_dial_powergrid_1.0'
        default_config_dir = parent_dir +'/algorithms/envs/PowerGrid/configs/config_ma2c_dial_DER6.ini'
    elif env_name == "PowerGrid" and algo_name == "NeurComm":
        default_base_dir = current_dir +'/ma2c_nc_powergrid_1.0'
        default_config_dir = parent_dir +'/algorithms/envs/PowerGrid/configs/config_ma2c_nc_DER6.ini'
    elif env_name == "PowerGrid" and algo_name == "ConseNet":
        default_base_dir = current_dir +'/ia2c_cu_powergrid_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ia2c_cu_DER6.ini'
    elif env_name == "PowerGrid" and algo_name == "FPrint":
        default_base_dir = current_dir +'/ia2c_fp_powergrid_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ia2c_fp_DER6.ini'



    elif env_name == "Pandemic" and algo_name == "CommNet":
        default_base_dir = current_dir +'/ma2c_cnet_Pandemic_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ma2c_cnet_DER6.ini'
    elif env_name == "Pandemic" and algo_name == "DIAL":
        default_base_dir = current_dir +'/ma2c_dial_Pandemic_1.0'
        default_config_dir = parent_dir +  '/algorithms/envs/PowerGrid/configs/config_ma2c_dial_DER6.ini'
    elif env_name == "Pandemic" and algo_name == "NeurComm":
        default_base_dir = current_dir +'/ma2c_nc_Pandemic_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ma2c_nc_DER6.ini'
    elif env_name == "Pandemic" and algo_name == "ConseNet":
        default_base_dir = current_dir +'/ia2c_cu_Pandemic_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ia2c_cu_DER6.ini'
    elif env_name == "Pandemic" and algo_name == "FPrint":
        default_base_dir = current_dir +'/ia2c_fp_Pandemic_1.0'
        default_config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ia2c_fp_DER6.ini'


    elif env_name == "Large_city" and algo_name == "CommNet":
        default_base_dir = current_dir +'/ma2c_cnet_Pandemic_1.0'
        default_config_dir = parent_dir +'/algorithms/envs/PowerGrid/configs/config_ma2c_cnet_DER6.ini'
    elif env_name == "Large_city" and algo_name == "DIAL":
        default_base_dir = current_dir +'/ma2c_dial_Pandemic_1.0'
        default_config_dir =parent_dir + '/algorithms/envs/PowerGrid/configs/config_ma2c_dial_DER6.ini'
    elif env_name == "Large_city" and algo_name == "NeurComm":
        default_base_dir = current_dir +'/ma2c_nc_Pandemic_1.0'
        default_config_dir = parent_dir +'/algorithms/envs/PowerGrid/configs/config_ma2c_nc_DER6.ini'
    elif env_name == "Large_city" and algo_name == "ConseNet":
        default_base_dir = current_dir +'/ia2c_cu_Pandemic_1.0'
        default_config_dir =parent_dir + '/algorithms/envs/PowerGrid/configs/config_ia2c_cu_DER6.ini'
    elif env_name == "Large_city" and algo_name == "FPrint":
        default_base_dir = current_dir +'/ia2c_fp_Pandemic_1.0'
        default_config_dir = parent_dir +'/algorithms/envs/PowerGrid/configs/config_ia2c_fp_DER6.ini'

    
    #default_base_dir = '/home/chengdong/deeprl_network/ma2c_nc_slowdown_1.0'
    #default_config_dir = './config/config_ma2c_nc_slowdown.ini'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    subparsers = parser.add_subparsers(dest='option', help="train or evaluate")
    sp = subparsers.add_parser('train', help='train a single agent under base dir')
    sp.add_argument('--config-dir', type=str, required=False,
                    default=default_config_dir, help="experiment config path")
    sp = subparsers.add_parser('evaluate', help="evaluate and compare agents under base dir")
    sp.add_argument('--evaluation-seeds', type=str, required=False,
                    default=','.join([str(i) for i in range(2000, 2500, 10)]),
                    help="random seeds for evaluation, split by ,")
    sp.add_argument('--demo', action='store_true', help="shows SUMO gui")
    args = parser.parse_args()
    if not args.option:
        parser.print_help()
        exit(1)
    return args


def init_env(config, port=0):
    scenario = config.get('scenario')
    if scenario.startswith('atsc'):
        if scenario.endswith('large_grid'):
            return LargeGridEnv(config, port=port)
        else:
            return RealNetEnv(config, port=port)
    elif scenario.startswith('cacc'):
        return CACCEnv(config)
    else:
        return CACCEnv(config)


def init_agent(env, config, total_step, seed):
    if  algo_name == "IA2C":

        if env_name == "Pandemic":
            n_s_ls = [16]*10
            n_a_ls = [5]*10
            neighbor_mask = np.eye(10)
            distance_mask = np.ones((10,10)) - np.eye(10)
            coop_gamma = 0.9
            return IA2C(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                           total_step, config, seed=seed)
        elif env_name == "Large_city":
            coop_gamma = 0.9
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            return IA2C(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                        total_step, config, seed=seed)

        return IA2C(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                    total_step, config, seed=seed)
                    
    elif algo_name == "FPrint":

        if env_name == "Pandemic":
            n_s_ls = [16]*10
            n_a_ls = [5]*10
            neighbor_mask = np.eye(10)
            distance_mask = np.ones((10,10)) - np.eye(10)
            coop_gamma = 0.9
            return IA2C(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                           total_step, config, seed=seed)

        # if env_name == "Pandemic":
        #     n_s_ls = [16]*10
        #     n_a_ls = [5]*10
        #     #neighbor_mask = np.eye(10)
            
        #     neighbor_mask = np.zeros((10, 10))
        #     n = 10
        #     for i in range(n):
        #         neighbor_mask[i][i] = 1
        #         neighbor_mask[i][(i+1)%n] = 1
        #         neighbor_mask[i][(i+n-1)%n] = 1

            
        #     neighbor_mask = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        #                                [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        #                                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        #                                [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
        #                                [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
        #                                [0, 0, 0, 0, 1, 0, 1, 0, 0, 0],
        #                                [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
        #                                [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
        #                                [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
        #                                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
            
        #     neighbor_mask = neighbor_mask + np.eye(10)

        #     print("neighbor_mask=",neighbor_mask)


        #     n_s_ls = []

               
            
        #     for i in range(10):
        #         n_n = np.sum(neighbor_mask[i])
        #         n_s_ls.append(16 * (n_n-1))

        #     n_s_ls = [32,48,48,48,48,48,48,48,48,32]

        #     distance_mask = neighbor_mask
        #     distance_mask = np.array( [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        #                                  [1, 0, 1, 2, 3, 4, 5, 6, 7, 8],
        #                                  [2, 1, 0, 1, 2, 3, 4, 5, 6, 7],
        #                                  [3, 2, 1, 0, 1, 2, 3, 4, 5, 6],
        #                                  [4, 3, 2, 1, 0, 1, 2, 3, 4, 5],
        #                                  [5, 4, 3, 2, 1, 0, 1, 2, 3, 4],
        #                                  [6, 5, 4, 3, 2, 1, 0, 1, 2, 3],
        #                                  [7, 6, 5, 4, 3, 2, 1, 0, 1, 2],
        #                                  [8, 7, 6, 5, 4, 3, 2, 1, 0, 1],
        #                                  [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]])
        #     coop_gamma = -1



        #     return IA2C_FP(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
        #                    total_step, config, seed=seed)

        elif env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 

            n_s_ls = []
            for i in range(env.n_agent):
                n_n = np.sum(neighbor_mask[i])
                n_s_ls.append(env.n_s * (n_n+1))
            
            return IA2C_FP(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                        total_step, config, seed=seed)
              
        # print("neighbor_mask =",env.neighbor_mask )
        # print("distance_mask =",env.distance_mask )
        # print("env.ns_ls=",env.n_s_ls)
        # print("env.na_ls=",env.n_a_ls)
        # print("env.coop_gamma=",env.coop_gamma)


        return IA2C_FP(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
                       
    elif algo_name == "NeurComm":
        if env_name == "Pandemic":
            n_s_ls = [16]*10
            n_a_ls = [5]*10
            neighbor_mask = np.eye(10)
            distance_mask = np.ones((10,10)) - np.eye(10)
            coop_gamma = -1
            return MA2C_NC(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                           total_step, config, seed=seed)
        elif env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            return MA2C_NC(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                        total_step, config, seed=seed)
        return MA2C_NC(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
                       
    elif algo_name == "CommNet":

        if env_name == "Pandemic":
            n_s_ls = [16]*10
            n_a_ls = [5]*10
            neighbor_mask = np.eye(10)
            distance_mask = np.ones((10,10)) - np.eye(10)
            coop_gamma = -1
            return MA2C_CNET(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                           total_step, config, seed=seed)
        elif env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            return MA2C_CNET(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                            total_step, config, seed=seed)

        # this is actually CommNet
        return MA2C_CNET(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
                         
    elif algo_name == "ConseNet":

        #print("env.n_s_ls=",env.n_s_ls)
        if env_name == "Monaco":
            n_s_ls = [22]*28
            return IA2C_CU(n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                           total_step, config, seed=seed)

        elif env_name == "Pandemic":
            n_s_ls = [16]*10
            n_a_ls = [5]*10
            neighbor_mask = np.eye(10)
            distance_mask = np.ones((10,10)) - np.eye(10)
            coop_gamma = -1
            return IA2C_CU(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                           total_step, config, seed=seed)
        elif env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            return IA2C_CU(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                        total_step, config, seed=seed)
            
        print("env.distance_mask=",env.distance_mask) 

        return IA2C_CU(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                       total_step, config, seed=seed)
                       
    elif algo_name == "DIAL":

        if env_name == "Monaco":
            n_s_ls = [22]*28
            return MA2C_DIAL(n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                           total_step, config, seed=seed)
        elif env_name == "Pandemic":
            n_s_ls = [16]*10
            n_a_ls = [5]*10
            neighbor_mask = np.eye(10)
            distance_mask = np.ones((10,10)) - np.eye(10)
            coop_gamma = -1
            return MA2C_DIAL(n_s_ls, n_a_ls, neighbor_mask, distance_mask, coop_gamma,
                           total_step, config, seed=seed)
        elif env_name == "Large_city":
            coop_gamma = -1
            adj_order = 30
            neighbor_mask = env.cal_n_order_matrix(env.n_agent,adj_order,env.neighbor_mask) 
            return MA2C_DIAL(env.n_s_ls, env.n_a_ls, neighbor_mask, env.distance_mask, coop_gamma,
                            total_step, config, seed=seed)
        return MA2C_DIAL(env.n_s_ls, env.n_a_ls, env.neighbor_mask, env.distance_mask, env.coop_gamma,
                         total_step, config, seed=seed)
    else:
        return None


def train(args, env_name):
    base_dir = args.base_dir
    dirs = init_dir(base_dir)
    init_log(dirs['log'])
    config_dir = args.config_dir
    copy_file(config_dir, dirs['data'])
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    #env = init_env(config['ENV_CONFIG'])
    
    

    alg_args = Config()
    
    if env_name == "catchup":
        env = CACC_catchup_2(config['ENV_CONFIG'])
        alg_args.env_fn = CACC_catchup_2  
    elif env_name == "slowdown":
        env = CACC_slowdown_2(config['ENV_CONFIG'])
        alg_args.env_fn = CACC_slowdown_2  
    elif env_name == "Grid":
        env = Grid_Env_2(config['ENV_CONFIG'])
        alg_args.env_fn = Grid_Env_2  
    elif env_name == "Monaco":
        env = Monaco_Env_2(config['ENV_CONFIG'])
        alg_args.env_fn = Monaco_Env_2
    elif env_name == "PowerGrid":
        env = GridEnv_2(config['ENV_CONFIG'])
        alg_args.env_fn = PowerGrid_Env_2
    elif env_name == "Pandemic":
        env = Pandemic_2()
        alg_args.env_fn = Pandemic_2
    elif env_name == "Large_city":
        env = Large_city_Env()
        alg_args.env_fn = Large_city_Env 

      
    alg_args.env_name = env_name
    alg_args.algo_name = algo_name

    run_args = Config()
    run_args.radius_v = 1
    run_args.radius_pi = 1
    run_args.radius_p = 1 
    run_args.debug = False
    run_args.name = run_args.name = env_name + "_" + algo_name
    run_args.save_period = 1800
    run_args.log_period = int(20)
    run_args.seed = int(time.time()*1000)%65536

    logger = LogServer({'run_args':run_args, 'algo_args':alg_args}, mute=run_args.debug)
    logger = LogClient(logger)
      



        
        
    #logging.info('Training: a dim %r, agent dim: %d' % (env.n_a_ls, env.n_agent))

    # init step counter
    total_step = int(config.getfloat('TRAIN_CONFIG', 'total_step'))
    test_step = int(config.getfloat('TRAIN_CONFIG', 'test_interval'))
    log_step = int(config.getfloat('TRAIN_CONFIG', 'log_interval'))
    global_counter = Counter(total_step, test_step, log_step)

    # init centralized or multi agent
    #seed = config.getint('ENV_CONFIG', 'seed')
    seed = random.randint(0,10000)
    seed = int(time.time()*1000)%65536
    model = init_agent(env, config['MODEL_CONFIG'], total_step, seed)
    model.load(dirs['model'], train_mode=True)

    # disable multi-threading for safe SUMO implementation
    summary_writer = SummaryWriter(dirs['log'], flush_secs=10000)
    trainer = Trainer(env, env_name,  algo_name, model, global_counter, summary_writer, output_path=dirs['data'], logger=logger)
    trainer.run()

    # save model
    final_step = global_counter.cur_step
    model.save(dirs['model'], final_step)
    summary_writer.close()


def evaluate_fn(agent_dir, output_dir, seeds, port, demo):
    agent = agent_dir.split('/')[-1]
    if not check_dir(agent_dir):
        logging.error('Evaluation: %s does not exist!' % agent)
        return
    # load config file 
    config_dir = find_file(agent_dir + '/data/')
    if not config_dir:
        return
    config = configparser.ConfigParser()
    config.read(config_dir)

    # init env
    env = init_env(config['ENV_CONFIG'], port=port)
    env.init_test_seeds(seeds)

    # load model for agent
    model = init_agent(env, config['MODEL_CONFIG'], 0, 0)
    if model is None:
        return
    model_dir = agent_dir + '/model/'
    if not model.load(model_dir):
        return
    # collect evaluation data
    evaluator = Evaluator(env, model, output_dir, gui=demo)
    evaluator.run()


def evaluate(args):
    base_dir = args.base_dir
    if not args.demo:
        dirs = init_dir(base_dir, pathes=['eva_data', 'eva_log'])
        init_log(dirs['eva_log'])
        output_dir = dirs['eva_data']
    else:
        output_dir = None
    # enforce the same evaluation seeds across agents
    seeds = args.evaluation_seeds
    logging.info('Evaluation: random seeds: %s' % seeds)
    if not seeds:
        seeds = []
    else:
        seeds = [int(s) for s in seeds.split(',')]
    evaluate_fn(base_dir, output_dir, seeds, 1, args.demo)


if __name__ == '__main__':
    args = parse_args(env_name, algo_name)
    if args.option == 'train':
        train(args, env_name)
    else:
        evaluate(args)

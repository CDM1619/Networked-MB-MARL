import configparser
from .PowerGrid.envs.Grid_envs import GridEnv
import os
from shutil import copy
import numpy as np

import sys
#sys.path.append("/home/chengdong/MARL/")

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
#print("current_dir=",current_dir)
# print("parent_dir=",parent_dir)
sys.path.append(parent_dir)


def copy_file(src_dir, tar_dir):
    copy(src_dir, tar_dir)
    env = parent_dir + '/algorithms/envs/PowerGrid/envs/Grid_envs.py'
    copy(env, tar_dir)
    policies = parent_dir + '/algorithms/envs/PowerGrid/agents/policies.py'
    copy(policies, tar_dir)
    models = parent_dir + '/algorithms/envs/PowerGrid/agents/models.py'
    copy(models, tar_dir)
    main = parent_dir + '/algorithms/envs/PowerGrid/main.py'
    copy(main, tar_dir)

def init_dir(base_dir, pathes=['log', 'data', 'model']):
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    dirs = {}
    for path in pathes:
        cur_dir = base_dir + '/%s/' % path
        if not os.path.exists(cur_dir):
            os.mkdir(cur_dir)
        dirs[path] = cur_dir
    return dirs


base_dir = './ma2c_cnet_der6'
dirs = init_dir(base_dir)
#config_dir = 'algorithms/envs/PowerGrid/configs/config_ma2c_cnet_DER6.ini'
config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ma2c_cnet_DER6.ini'
copy_file(config_dir, dirs['data'])
config = configparser.ConfigParser()
config.read(config_dir)
seed = config.getint('ENV_CONFIG', 'seed')



env = GridEnv(config['ENV_CONFIG'], random_seed=seed)
env.train_mode = True


def PowerGrid_Env():
    return env

def PowerGrid_Env_2():
    return env




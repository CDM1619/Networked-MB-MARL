from configs.Config import Config
from env.flatland.EnvCurriculum import EnvCurriculum, EnvCurriculumSample, EnvCurriculumPrioritizedSample
# from env.flatland.Flatland import FlatlandWrapper, Flatland
# from env.flatland.GreedyFlatland import GreedyFlatland
#from env.starcraft.StarCraft import StarCraft

import os
import sys
import time
import importlib
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(parent_dir)
from algorithms.envs.CACC import CACC_catchup_3, CACC_slowdown_3, CACC_catchup_test, CACC_slowdown_test
from algorithms.envs.ATSC import Grid_Env_3, Monaco_Env_3
from algorithms.envs.PowerGrid.envs.Grid_envs import GridEnv_3
import configparser
from algorithms.utils import Config, LogClient, LogServer, mem_report

class EnvConfig(Config):
    def __init__(self):
        pass

    def create_env(self):
        pass


class StarCraftConfig(EnvConfig):

    def __init__(self, env_name):
        self.env_name = env_name

    def create_env(self):
        return StarCraft(self.env_name)


class CACCConfig(EnvConfig):

    def __init__(self, env_name):
        self.env_name = env_name


    def create_env(self):
        if self.env_name == "catchup":
            config_dir = parent_dir + '/algorithms/envs/NCS/config/config_ma2c_nc_catchup.ini'
            config = configparser.ConfigParser()
            config.read(config_dir)
            return CACC_catchup_3(config['ENV_CONFIG'])
        elif self.env_name == "slowdown":
            config_dir = parent_dir + '/algorithms/envs/NCS/config/config_ma2c_nc_slowdown.ini'
            config = configparser.ConfigParser()
            config.read(config_dir)
            return CACC_slowdown_3(config['ENV_CONFIG'])

class ATSCConfig(EnvConfig):

    def __init__(self, env_name):
        self.env_name = env_name


    def create_env(self):
        if self.env_name == "Grid":
            config_dir = parent_dir + "/algorithms/envs/NCS/config/config_ma2c_nc_grid.ini"
            config = configparser.ConfigParser()
            config.read(config_dir)
            return Grid_Env_3(config['ENV_CONFIG'])
        elif self.env_name == "Monaco":
            config_dir = parent_dir + "/algorithms/envs/NCS/config/config_ma2c_nc_net.ini"
            config = configparser.ConfigParser()
            config.read(config_dir)
            return Monaco_Env_3(config['ENV_CONFIG'])


class PowerGridConfig(EnvConfig):

    def __init__(self, env_name):
        self.env_name = env_name


    def create_env(self):

        config_dir = parent_dir + '/algorithms/envs/PowerGrid/configs/config_ma2c_cnet_DER6.ini'
        config = configparser.ConfigParser()
        config.read(config_dir)
        return GridEnv_3(config['ENV_CONFIG'])




class FlatlandConfig(EnvConfig):
    def __init__(self,
                 height,
                 width,
                 n_agents,
                 n_cities,
                 grid_distribution_of_cities,
                 max_rails_between_cities,
                 max_rail_in_cities,
                 observation_builder_config,
                 reward_config,
                 malfunction_rate,
                 greedy,
                 random_seed):
        super(FlatlandConfig, self).__init__()
        self.height = height
        self.width = width
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.grid_distribution_of_cities = grid_distribution_of_cities
        self.max_rails_between_cities = max_rails_between_cities
        self.max_rail_in_cities = max_rail_in_cities
        self.observation_builder_config = observation_builder_config
        self.reward_config = reward_config
        self.malfunction_rate = malfunction_rate
        self.random_seed = random_seed
        self.greedy = greedy

    def update_random_seed(self):
        self.random_seed += 1

    def set_obs_builder_config(self, obs_builder_config):
        self.observation_builder_config = obs_builder_config

    def set_reward_config(self, reward_config):
        self.reward_config = reward_config

    def create_env(self):
        obs_builder = self.observation_builder_config.create_builder()
        reward_shaper = self.reward_config.create_reward_shaper()
        rail_env = FlatlandWrapper(Flatland(height=self.height,
                                            width=self.width,
                                            n_agents=self.n_agents,
                                            n_cities=self.n_cities,
                                            grid_distribution_of_cities=self.grid_distribution_of_cities,
                                            max_rails_between_cities=self.max_rails_between_cities,
                                            max_rail_in_cities=self.max_rail_in_cities,
                                            observation_builder=obs_builder,
                                            malfunction_rate=self.malfunction_rate,
                                            random_seed=self.random_seed),
                                   reward_shaper=reward_shaper)
        if self.greedy:
            rail_env = GreedyFlatland(rail_env)
        return rail_env


class EnvCurriculumConfig(EnvConfig):
    def __init__(self, env_configs, env_episodes, env_type, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_episodes = env_episodes
        self.ENV_TYPE = env_type

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculum(self.env_configs, self.env_episodes)


class EnvCurriculumSampleConfig(EnvConfig):
    def __init__(self, env_configs, env_probs, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.env_probs = env_probs

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumSample(self.env_configs, self.env_probs)


class EnvCurriculumPrioritizedSampleConfig(EnvConfig):
    def __init__(self, env_configs, repeat_random_seed, obs_builder_config=None, reward_config=None):
        self.env_configs = env_configs
        self.repeat_random_seed = repeat_random_seed

        if obs_builder_config is not None:
            self.set_obs_builder_config(obs_builder_config)

        if reward_config is not None:
            self.set_reward_config(reward_config)

    def update_random_seed(self):
        for conf in self.env_configs:
            conf.update_random_seed()

    def set_obs_builder_config(self, obs_builder_config):
        for conf in self.env_configs:
            conf.set_obs_builder_config(obs_builder_config)

    def set_reward_config(self, reward_config):
        for conf in self.env_configs:
            conf.set_reward_config(reward_config)

    def create_env(self):
        return EnvCurriculumPrioritizedSample(self.env_configs, self.repeat_random_seed)

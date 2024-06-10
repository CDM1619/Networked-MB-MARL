from algorithms.envs.flow import envs
from ..envs.flow.envs.ring.accel import AccelEnv
from ..envs.flow.networks import FigureEightNetwork
from ..envs.flow.core import rewards
from copy import deepcopy
from algorithms.envs.flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, \
    SumoCarFollowingParams
from algorithms.envs.flow.core.params import VehicleParams
from algorithms.envs.flow.controllers import IDMController, ContinuousRouter, RLController
from algorithms.envs.flow.core.params import TrafficLightParams
from algorithms.envs.flow.networks.figure_eight import ADDITIONAL_NET_PARAMS
import gym
from gym.spaces import Box, Discrete
from gym.envs.registration import register
import numpy as np

from ..envs.flow.networks import RingNetwork

class FigureEightWrapper(AccelEnv):
    def __init__(self, env_params, sim_params, network, simulator):
        super().__init__(env_params, sim_params, network, simulator=simulator)
        self.n_agent = self.initial_vehicles.num_vehicles
        self.n_s_ls, self.n_a_ls, self.coop_gamma, self.distance_mask, self.neighbor_mask \
            = [], [], -1, np.zeros((self.n_agent, self.n_agent)), np.zeros((self.n_agent, self.n_agent))
        self.init_neighbor_mask()
        self.init_distance_mask()
        self.n_s_ls = [2] * self.n_agent
        self.n_a_ls = [1] * self.n_agent
        
        #self.neighbor_mask = np.identity(self.n_agent)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity', 'Absolute_pos']
        return Box(
            low=0,
            high=1,
            shape=(2, ),
            dtype=np.float32)

    def get_state_(self):
        """See class definition."""
        
        # for veh_id in self.sorted_ids:
            # print('speed=',self.k.network.max_speed())
            # print('po=',self.k.network.length())

        speed = [self.k.vehicle.get_speed(veh_id) / self.k.network.max_speed()
                 for veh_id in self.sorted_ids]
        pos = [self.k.vehicle.get_x_by_id(veh_id) / self.k.network.length()
               for veh_id in self.sorted_ids]
        speed = np.array(speed).reshape((-1, 1))
        pos = np.array(pos).reshape((-1, 1))

        return np.concatenate([speed, pos], axis=-1)
    
    def step(self, rl_actions: np.array):
        while rl_actions.ndim > 1:
            rl_actions = rl_actions.squeeze(-1)
        _, _, d, info = super().step(rl_actions)
        s1 = self.get_state_()
        r = self.get_reward_()
        d = np.array([d] * self.n_agent, dtype=np.bool)
        return s1, r, d, info
        
    
    def get_reward_(self):
        if self.env_params.evaluate:
            return np.mean(self.k.vehicle.get_speed(self.sorted_ids))
        else:
            veh_ids = self.sorted_ids
            vel = np.array(self.k.vehicle.get_speed(veh_ids))
            target_vel = self.env_params.additional_params['target_velocity']
            return target_vel - np.abs(target_vel - vel)

    def _comparable_reward(self):
        comp_r = self.compute_reward(None, fail=False)
        n = self.n_agent
        return np.array([comp_r / n] * n, dtype=np.float32)

    def init_neighbor_mask(self):
        n = self.n_agent
        for i in range(n):
            self.neighbor_mask[i][i] = 1
            self.neighbor_mask[i][(i+1)%n] = 1
            self.neighbor_mask[i][(i+n-1)%n] = 1

    def init_distance_mask(self):
        n = self.n_agent
        for i in range(n):
            for j in range(n):
                self.distance_mask[i][j] = min((i-j+n)%n, (j-i+n)%n)
    
    def rescaleReward(self, ep_return, ep_len):
        return ep_return
    
    # def reset(self):
    #     return super().reset()
        

def makeFigureEight2(evaluate=False, version=0, render=None):
    HORIZON = 1500
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        car_following_params=SumoCarFollowingParams(
            speed_mode="obey_safe_speed",
        ),
        num_vehicles=14)

    flow_params = dict(
        # name of the experiment
        exp_tag="figure_eight_2",

        # name of the flow environment the experiment is running on
        env_name=FigureEightWrapper,

        # name of the network class the experiment is running on
        network=FigureEightNetwork,


        # simulator that is used by the experiment
        simulator='traci',

        # sumo-related parameters (see flow.core.params.SumoParams)
        sim=SumoParams(
            sim_step=0.1,
            render=render,
            no_step_log=True,
            print_warnings=False
        ),

        # environment related parameters (see flow.core.params.EnvParams)
        env=EnvParams(
            horizon=HORIZON,
            additional_params={
                "target_velocity": 20,
                "max_accel": 3,
                "max_decel": 3,
                "sort_vehicles": False
            },
        ),

        # network-related parameters (see flow.core.params.NetParams and the
        # network's documentation or ADDITIONAL_NET_PARAMS component)
        net=NetParams(
            additional_params=deepcopy(ADDITIONAL_NET_PARAMS),
        ),

        # vehicles to be placed in the network at the start of a rollout (see
        # flow.core.params.VehicleParams)
        veh=vehicles,

        # parameters specifying the positioning of vehicles upon initialization/
        # reset (see flow.core.params.InitialConfig)
        initial=InitialConfig(),
    )

    params = flow_params
    exp_tag = params["exp_tag"]
    base_env_name = params["env_name"].__name__

    # deal with multiple environments being created under the same name
    all_envs = gym.envs.registry.all()
    env_ids = [env_spec.id for env_spec in all_envs]
    while "{}-v{}".format(base_env_name, version) in env_ids:
        version += 1
    env_name = "{}-v{}".format(base_env_name, version)
    network_class = params["network"]

    env_params = params['env']
    net_params = params['net']
    initial_config = params.get('initial', InitialConfig())
    traffic_lights = params.get("tls", TrafficLightParams())
    sim_params = deepcopy(params['sim'])
    vehicles = deepcopy(params['veh'])

    network = network_class(
        name=exp_tag,
        vehicles=vehicles,
        net_params=net_params,
        initial_config=initial_config,
        traffic_lights=traffic_lights,
    )

    # accept new render type if not set to None
    sim_params.render = render or sim_params.render

    entry_point = params["env_name"].__module__ + ':' + params["env_name"].__name__

    # register the environment with OpenAI gym
    register(
        id=env_name,
        entry_point=entry_point,
        kwargs={
            "env_params": env_params,
            "sim_params": sim_params,
            "network": network,
            "simulator": params['simulator']
        })

    return gym.envs.make(env_name)

def makeFigureEightTest():
    return makeFigureEight2(evaluate=True)

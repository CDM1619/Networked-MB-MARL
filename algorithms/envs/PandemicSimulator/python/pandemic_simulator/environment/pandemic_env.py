# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.
from typing import List, Optional, Dict, Tuple, Mapping, Type, Sequence

import gym
import numpy as np
from gym.spaces import Box, Discrete

from .done import DoneFunction
from .interfaces import LocationID, PandemicObservation, NonEssentialBusinessLocationState, PandemicRegulation, \
    InfectionSummary
from .pandemic_sim import PandemicSim
from .reward import RewardFunction, SumReward, RewardFunctionFactory, RewardFunctionType
from .simulator_config import PandemicSimConfig
from .simulator_opts import PandemicSimOpts

__all__ = ['PandemicGymEnv']


def check_list_items_not_equal(lst, a):
    for item in lst:
        if item == a:
            return False  
    return True  

class PandemicGymEnv(gym.Env):
    """A gym environment interface wrapper for the Pandemic Simulator."""

    _pandemic_sim: PandemicSim
    _stage_to_regulation: Mapping[int, PandemicRegulation]
    _obs_history_size: int
    _sim_steps_per_regulation: int
    _non_essential_business_loc_ids: Optional[List[LocationID]]
    _reward_fn: Optional[RewardFunction]
    _done_fn: Optional[DoneFunction]

    _last_observation: PandemicObservation
    _last_reward: float

    def __init__(self,
                 Nums_Location,
                 pandemic_sim: PandemicSim,
                 pandemic_regulations: Sequence[PandemicRegulation],
                 reward_fn: Optional[RewardFunction] = None,
                 done_fn: Optional[DoneFunction] = None,
                 obs_history_size: int = 1,
                 sim_steps_per_regulation: int = 24,
                 non_essential_business_location_ids: Optional[List[LocationID]] = None,
                 ):
        """
        :param pandemic_sim: Pandemic simulator instance
        :param pandemic_regulations: A sequence of pandemic regulations
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param sim_steps_per_regulation: number of sim_steps to run for each regulation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        self.n_agent = 10
        self.n_agents = 10
        self.n_a = 5
        self.n_s = 16
        #self.n_s_ls = [16]*10
        #self.n_a_ls = [5]*10
        self.action_space = Discrete(5)
        self.fp = np.ones((self.n_agent, self.n_a)) / self.n_a
        self.coop_gamma = -1
        self.T = 120

        print("999999999999999999999999999999999999999999999999999999999999999999999999999999999999999999")
        
        self.Nums_Location = Nums_Location
        self._pandemic_sim = pandemic_sim
        self._stage_to_regulation = {reg.stage: reg for reg in pandemic_regulations}
        self._obs_history_size = obs_history_size
        self._sim_steps_per_regulation = sim_steps_per_regulation
        self.neighbor_mask = np.eye(10)
        


        if non_essential_business_location_ids is not None:
            for loc_id in non_essential_business_location_ids:
                assert isinstance(self._pandemic_sim.state.id_to_location_state[loc_id],
                                  NonEssentialBusinessLocationState)
        self._non_essential_business_loc_ids = non_essential_business_location_ids

        self._reward_fn = reward_fn
        self._done_fn = done_fn

        self.action_space = gym.spaces.Discrete(len(self._stage_to_regulation))

    @classmethod
    def from_config(cls: Type['PandemicGymEnv'],
                    Nums_Location,
                    sim_config: PandemicSimConfig,
                    pandemic_regulations: Sequence[PandemicRegulation],
                    sim_opts: PandemicSimOpts = PandemicSimOpts(),
                    reward_fn: Optional[RewardFunction] = None,
                    done_fn: Optional[DoneFunction] = None,
                    obs_history_size: int = 1,
                    non_essential_business_location_ids: Optional[List[LocationID]] = None,
                    ) -> 'PandemicGymEnv':
        """
        Creates an instance using config

        :param sim_config: Simulator config
        :param pandemic_regulations: A sequence of pandemic regulations
        :param sim_opts: Simulator opts
        :param reward_fn: reward function
        :param done_fn: done function
        :param obs_history_size: number of latest sim step states to include in the observation
        :param non_essential_business_location_ids: an ordered list of non-essential business location ids
        """
        sim = PandemicSim.from_config(sim_config, sim_opts)
        
        
        

        if sim_config.max_hospital_capacity == -1:
            raise Exception("Nothing much to optimise if max hospital capacity is -1.")

        reward_fn = reward_fn or SumReward(
            reward_fns=[
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              summary_type=InfectionSummary.CRITICAL,
                                              threshold=3 * sim_config.max_hospital_capacity),
                RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              num_stages=len(pandemic_regulations)),
                RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              num_stages=len(pandemic_regulations))
            ],
            weights=[.4, 1, .1, 0.02]
        )


        #reward_fn = reward_fn or SumReward(
            #reward_fns=[
                #RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_INCREASE,
                                              #summary_type=InfectionSummary.CRITICAL),
                #RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD,
                                              #summary_type=InfectionSummary.CRITICAL,
                                              #threshold=3 * sim_config.max_hospital_capacity),
                #RewardFunctionFactory.default(RewardFunctionType.LOWER_STAGE,
                                              #num_stages=len(pandemic_regulations)),
                #RewardFunctionFactory.default(RewardFunctionType.SMOOTH_STAGE_CHANGES,
                                              #num_stages=len(pandemic_regulations)),
                #RewardFunctionFactory.default(RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE,
                                              #summary_type=InfectionSummary.CRITICAL),
                #RewardFunctionFactory.default(RewardFunctionType.UNLOCKED_BUSINESS_LOCATIONS,
                                              #obs_indices=None),
            #],
            #weights=[.4, 1, .1, 0.02, 0.5, 0.5]
        #)

        return PandemicGymEnv(Nums_Location,
                              pandemic_sim=sim,
                              pandemic_regulations=pandemic_regulations,
                              sim_steps_per_regulation=sim_opts.sim_steps_per_regulation,
                              reward_fn=reward_fn,
                              done_fn=done_fn,
                              obs_history_size=obs_history_size,
                              non_essential_business_location_ids=non_essential_business_location_ids)

    @property
    def pandemic_sim(self) -> PandemicSim:
        return self._pandemic_sim

    @property
    def observation(self) -> PandemicObservation:
        return self._last_observation

    @property
    def last_reward(self) -> float:
        return self._last_reward

    def step(self, action: int) -> Tuple[PandemicObservation, float, bool, Dict]:
        #assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        
        #print("action=",action)

        
        #print("self._last_observation.stage[-1, 0, 0]=",self._last_observation.stage[-1, 0, 0])
        

        #if check_list_items_not_equal(action, self._last_observation.stage[-1, 0, 0]):

        regulation = [self._stage_to_regulation[action[i]] for i in range(len(action))]
        self._pandemic_sim.impose_regulation(regulation=regulation)

        # update the sim until next regulation interval trigger and construct obs from state hist
        obs = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)

        hist_index = 0
        

        
        #print("self._pandemic_sim.state=",self._pandemic_sim.state.id_to_location_state)
        #print("1111111111111111111111111111111111=",self._pandemic_sim.state.id_to_location_state[LocationID(name='School_0')].contact_rate.min_assignees)

        #for key in self._pandemic_sim.state.id_to_location_state:
            #value = self._pandemic_sim.state.id_to_location_state[key]
            #print("key=",key)
            #print("value=",value)

        #for person_id in self._pandemic_sim.state.id_to_location_state:
            #print("person_id.name=",person_id.name)

        
        for i in range(self._sim_steps_per_regulation):
            # step sim
            self._pandemic_sim.step()

            # store only the last self._history_size state values
            if i >= (self._sim_steps_per_regulation - self._obs_history_size):
                obs.update_obs_with_sim_state(self._pandemic_sim.state, hist_index,
                                              self._non_essential_business_loc_ids)
                hist_index += 1

        prev_obs = self._last_observation
        self._last_reward = self._reward_fn.calculate_reward(prev_obs, action, obs) if self._reward_fn else 0.
        done = self._done_fn.calculate_done(obs, action) if self._done_fn else False
        self._last_observation = obs

        return self.get_state_(), np.array([self._last_reward]*self.n_agent), np.array([done]*self.n_agent), {}

    def get_neighbor_action(self, action):
        naction = []
        for i in range(self.n_agent):
            naction.append(action[self.neighbor_mask[i] == 1])
        return naction
    

    def get_state_(self):
        obs = np.concatenate((self._last_observation.global_infection_summary[0][0], self._last_observation.global_testing_summary[0][0], self._last_observation.stage[0][0], self._last_observation.infection_above_threshold[0][0], self._last_observation.time_day[0][0]))
        


        
        Home_state = []
        GroceryStore_state = []
        Office_state = []
        School_state = []
        Hospital_state = []
        RetailStore_state = []
        HairSalon_state = []
        Restaurant_state = []
        Bar_state = []
        Population_state = np.concatenate((np.array([0,0,0]),obs))
        for i in range(len(self.Nums_Location)):

            if i == 0:
                Home_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Home_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                Home_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Home_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                Home_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Home_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                Home_state = np.concatenate((np.array(Home_state),obs))
            if i == 1:
                GroceryStore_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='GroceryStore_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                GroceryStore_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='GroceryStore_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                GroceryStore_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='GroceryStore_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                GroceryStore_state = np.concatenate((np.array(GroceryStore_state),obs))
            if i == 2:
                Office_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Office_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                Office_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Office_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                Office_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Office_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                Office_state = np.concatenate((np.array(Office_state),obs))
            if i == 3:
                School_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='School_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                School_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='School_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                School_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='School_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                School_state = np.concatenate((np.array(School_state),obs))
            if i == 4:
                Hospital_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Hospital_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                Hospital_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Hospital_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                Hospital_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Hospital_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                Hospital_state = np.concatenate((np.array(Hospital_state),obs))
            if i == 5:
                RetailStore_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='RetailStore_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                RetailStore_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='RetailStore_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                RetailStore_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='RetailStore_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                RetailStore_state = np.concatenate((np.array(RetailStore_state),obs))
            if i == 6:
                HairSalon_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='HairSalon_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                HairSalon_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='HairSalon_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                HairSalon_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='HairSalon_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                HairSalon_state = np.concatenate((np.array(HairSalon_state),obs))
            if i == 7:
                Restaurant_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Restaurant_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                Restaurant_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Restaurant_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                Restaurant_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Restaurant_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                Restaurant_state = np.concatenate((np.array(Restaurant_state),obs))
            if i == 8:
                Bar_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Bar_'+str(j))].contact_rate.fraction_assignees for j in range(self.Nums_Location[i])]))
                Bar_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Bar_'+str(j))].contact_rate.fraction_assignees_visitors for j in range(self.Nums_Location[i])]))
                Bar_state.append(sum([self._pandemic_sim.state.id_to_location_state[LocationID(name='Bar_'+str(j))].contact_rate.fraction_visitors for j in range(self.Nums_Location[i])]))
                Bar_state = np.concatenate((np.array(Bar_state),obs))
        
                
        state = np.array([Home_state, GroceryStore_state, Office_state, School_state, Hospital_state, RetailStore_state, HairSalon_state, Restaurant_state, Bar_state, Population_state])
        return state
        
    def get_fingerprint(self):
        return self.fp

    def update_fingerprint(self, fp):
        self.fp = fp
    def reset(self) -> PandemicObservation:
        self._pandemic_sim.reset()

        

        self._last_observation = PandemicObservation.create_empty(
            history_size=self._obs_history_size,
            num_non_essential_business=len(self._non_essential_business_loc_ids)
            if self._non_essential_business_loc_ids is not None else None)
        
        #print("self._last_observationself._last_observation=",self._last_observation)
        #print("self._obs_history_size=",self._obs_history_size)
        self._last_reward = 0.0
        if self._done_fn is not None:
            self._done_fn.reset()
        return self._last_observation

    def render(self, mode: str = 'human') -> bool:
        pass

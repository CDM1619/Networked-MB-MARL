# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import enum
from abc import abstractmethod, ABCMeta
from typing import Any, Dict, List, Optional, Union, Type, Sequence

import numpy as np

__all__ = ['RewardFunction', 'RewardFunctionType', 'RewardFunctionFactory', 'SumReward',
           'UnlockedBusinessLocationsReward', 'InfectionSummaryIncreaseReward',
           'InfectionSummaryAboveThresholdReward', 'LowerStageReward', 'InfectionSummaryAbsoluteReward',
           'SmoothStageChangesReward']

from .interfaces import PandemicObservation, InfectionSummary, sorted_infection_summary


class RewardFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        pass


class RewardFunctionType(enum.Enum):
    INFECTION_SUMMARY_INCREASE = 'infection_summary_increase'
    INFECTION_SUMMARY_ABOVE_THRESHOLD = 'infection_summary_above_threshold'
    INFECTION_SUMMARY_ABSOLUTE = 'infection_summary_absolute'
    UNLOCKED_BUSINESS_LOCATIONS = 'unlocked_business_locations'
    LOWER_STAGE = 'lower_stage'
    SMOOTH_STAGE_CHANGES = 'smooth_stage_changes'

    @staticmethod
    def values() -> List[str]:
        return [c.value for c in RewardFunctionType.__members__.values()]


_REWARDS_REGISTRY: Dict[RewardFunctionType, Type[RewardFunction]] = {}


def _register_reward(type: RewardFunctionType, reward_fun: Type[RewardFunction]) -> None:
    if type not in _REWARDS_REGISTRY:
        _REWARDS_REGISTRY[type] = reward_fun
        return

    raise RuntimeError(f'Reward type {type} already registered')


class RewardFunctionFactory:
    @staticmethod
    def default(reward_function_type: Union[str, RewardFunctionType], *args: Any, **kwargs: Any) -> RewardFunction:
        rf_type = RewardFunctionType(reward_function_type)

        if rf_type not in _REWARDS_REGISTRY:
            raise ValueError('Unknown reward function type.')

        return _REWARDS_REGISTRY[rf_type](*args, **kwargs)


class SumReward(RewardFunction):
    """
    Reward function that sums the values of multiple reward functions.
    """
    _reward_fns: List[RewardFunction]
    _weights: np.ndarray

    def __init__(self, reward_fns: List[RewardFunction], weights: Optional[List[float]] = None,
                 *args: Any,
                 **kwargs: Any):
        """
        :param reward_fns: List of reward functions to sum.
        :param weights: Weights for each reward function. If None, each weight is set to 1.
        """
        super().__init__(*args, **kwargs)
        if weights is not None:
            assert len(weights) == len(reward_fns), 'There must be one weight for each reward function.'
        else:
            weights = [1.] * len(reward_fns)
        self._weights = np.asarray(weights)
        self._reward_fns = reward_fns

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        rewards = np.array([rf.calculate_reward(prev_obs, action, obs) for rf in self._reward_fns])
        return float(np.sum(rewards * self._weights))


class InfectionSummaryIncreaseReward(RewardFunction):
    """Returns a negative reward proportional to the relative increase in the infection summary of the given type."""
    _index: int

    def __init__(self, summary_type: InfectionSummary, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert summary_type in [InfectionSummary.INFECTED, InfectionSummary.CRITICAL, InfectionSummary.DEAD]
        self._index = sorted_infection_summary.index(summary_type)

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        prev_summary = prev_obs.global_infection_summary[..., self._index]
        summary = obs.global_infection_summary[..., self._index]
        if np.any(prev_summary == 0):
            return 0
        return -1 * float(np.clip((summary - prev_summary) / prev_summary, 0, np.inf).mean())


class InfectionSummaryAbsoluteReward(RewardFunction):
    """Returns a negative reward proportional to the absolute value of the given type of infection summary."""
    _index: int

    def __init__(self, summary_type: InfectionSummary, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        assert summary_type in [InfectionSummary.INFECTED, InfectionSummary.CRITICAL, InfectionSummary.DEAD]
        self._index = sorted_infection_summary.index(summary_type)

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        return float(-1 * np.mean(obs.global_infection_summary[..., self._index]))


class InfectionSummaryAboveThresholdReward(RewardFunction):
    """Returns a negative reward if the infection summary of the given type is above a threshold."""
    _threshold: float
    _index: int

    def __init__(self, summary_type: InfectionSummary, threshold: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        assert summary_type in [InfectionSummary.INFECTED, InfectionSummary.CRITICAL, InfectionSummary.DEAD]
        self._index = sorted_infection_summary.index(summary_type)

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        return float(-1 * max(np.mean(obs.global_infection_summary[..., self._index]
                                      - self._threshold) / self._threshold, 0))


class UnlockedBusinessLocationsReward(RewardFunction):
    """Returns a positive reward proportional to the number of unlocked business locations."""
    _obs_indices: Optional[Sequence[int]] = None

    def __init__(self, obs_indices: Optional[Sequence[int]] = None, *args: Any, **kwargs: Any):
        """
        :param obs_indices: indices of certain business locations in obs to use. If None, all business location ids
            are used.
        """
        super().__init__(*args, **kwargs)
        self._obs_indices = obs_indices

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        if obs.unlocked_non_essential_business_locations is None:
            return 0.
        else:
            unlocked_locations = (obs.unlocked_non_essential_business_locations if self._obs_indices is None
                                  else obs.unlocked_non_essential_business_locations[..., self._obs_indices])
            return float(np.mean(unlocked_locations))


class LowerStageReward(RewardFunction):
    """Returns a positive reward inversely proportional to the regulation stages."""
    _stage_rewards: np.ndarray

    def __init__(self, num_stages: int, *args: Any, **kwargs: Any):
        """
        :param num_stages: total number of stages
        """
        super().__init__(*args, **kwargs)
        stage_rewards = np.arange(0, num_stages) ** 1.5
        self._stage_rewards = stage_rewards / np.max(stage_rewards)

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        return -float(sum(self._stage_rewards[action[i]] for i in range(len(action))))


class SmoothStageChangesReward(RewardFunction):

    def __init__(self, num_stages: int, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def calculate_reward(self, prev_obs: PandemicObservation, action: int, obs: PandemicObservation) -> float:
        return float(-1 * np.abs(obs.stage - prev_obs.stage).mean())


_register_reward(RewardFunctionType.INFECTION_SUMMARY_INCREASE, InfectionSummaryIncreaseReward)
_register_reward(RewardFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD, InfectionSummaryAboveThresholdReward)
_register_reward(RewardFunctionType.INFECTION_SUMMARY_ABSOLUTE, InfectionSummaryAbsoluteReward)
_register_reward(RewardFunctionType.UNLOCKED_BUSINESS_LOCATIONS, UnlockedBusinessLocationsReward)
_register_reward(RewardFunctionType.LOWER_STAGE, LowerStageReward)
_register_reward(RewardFunctionType.SMOOTH_STAGE_CHANGES, SmoothStageChangesReward)

# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import enum
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Type, Union, Any

import numpy as np

__all__ = ['DoneFunctionType', 'DoneFunctionFactory', 'ORDone', 'DoneFunction',
           'InfectionSummaryAboveThresholdDone', 'NoMoreInfectionsDone', 'NoPandemicDone']

from .interfaces import PandemicObservation, InfectionSummary, sorted_infection_summary


class DoneFunction(metaclass=ABCMeta):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def calculate_done(self, obs: PandemicObservation, action: int) -> bool:
        pass

    def reset(self) -> None:
        pass


class DoneFunctionType(enum.Enum):
    INFECTION_SUMMARY_ABOVE_THRESHOLD = 'infection_summary_above_threshold'
    NO_MORE_INFECTIONS = 'no_more_infections'
    NO_PANDEMIC = 'no_pandemic'

    @staticmethod
    def values() -> List[str]:
        return [c.value for c in DoneFunctionType.__members__.values()]


_DONE_REGISTRY: Dict[DoneFunctionType, Type[DoneFunction]] = {}


def _register_done(type: DoneFunctionType, done_fun: Type[DoneFunction]) -> None:
    if type not in _DONE_REGISTRY:
        _DONE_REGISTRY[type] = done_fun
        return

    raise RuntimeError(f'Done type {type} already registered')


class DoneFunctionFactory:
    @staticmethod
    def default(done_function_type: Union[str, DoneFunctionType], *args: Any, **kwargs: Any) -> DoneFunction:
        df_type = DoneFunctionType(done_function_type)

        if df_type not in _DONE_REGISTRY:
            raise ValueError('Unknown done function type.')

        return _DONE_REGISTRY[df_type](*args, **kwargs)


class ORDone(DoneFunction):
    """
    Done function that takes logical OR of multiple done functions
    """

    _done_fns: List[DoneFunction]

    def __init__(self, done_fns: List[DoneFunction], *args: Any, **kwargs: Any):
        """
        :param done_fns: List of done functions to take logical OR.
        """
        super().__init__(*args, **kwargs)
        self._done_fns = done_fns

    def calculate_done(self, obs: PandemicObservation, action: int) -> bool:
        return any([df.calculate_done(obs, action) for df in self._done_fns])

    def reset(self) -> None:
        for done_fn in self._done_fns:
            done_fn.reset()


class InfectionSummaryAboveThresholdDone(DoneFunction):
    """Returns True if the infection summary of the given type is above a threshold."""
    _threshold: float
    _index: int

    def __init__(self, summary_type: InfectionSummary, threshold: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._threshold = threshold
        assert summary_type in [InfectionSummary.INFECTED, InfectionSummary.CRITICAL, InfectionSummary.DEAD]
        self._index = sorted_infection_summary.index(summary_type)

    def calculate_done(self, obs: PandemicObservation, action: int) -> bool:
        return bool(np.any(obs.global_infection_summary[..., self._index] > self._threshold))


class NoMoreInfectionsDone(DoneFunction):
    """Returns True if the number of infected and critical becomes zero and all have recovered."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._infected_index = sorted_infection_summary.index(InfectionSummary.INFECTED)
        self._recovered_index = sorted_infection_summary.index(InfectionSummary.RECOVERED)
        self._dead_index = sorted_infection_summary.index(InfectionSummary.DEAD)
        self._critical_index = sorted_infection_summary.index(InfectionSummary.CRITICAL)
        self._cnt = 0

    def calculate_done(self, obs: PandemicObservation, action: int) -> bool:
        no_infection = np.sum(obs.global_infection_summary[..., [self._infected_index, self._critical_index]]) == 0
        if no_infection and self._cnt > 5:
            return True
        elif no_infection:
            self._cnt += 1
        else:
            self._cnt = 0
        return False

    def reset(self) -> None:
        self._cnt = 0


class NoPandemicDone(DoneFunction):
    """Returns True if the pandemic hasn't started within the specified number of days."""

    _num_days: int
    _pandemic_exists: bool

    def __init__(self, num_days: int, *args: Any, **kwargs: Any):
        """
        :param num_days: number of days to check if pandemic has started
        """
        super().__init__(*args, **kwargs)
        self._pandemic_exists = False
        self._num_days = num_days

    def calculate_done(self, obs: PandemicObservation, action: int) -> bool:
        self._pandemic_exists = self._pandemic_exists or np.any(obs.infection_above_threshold)
        return obs.time_day[-1].item() > self._num_days and not self._pandemic_exists


_register_done(DoneFunctionType.INFECTION_SUMMARY_ABOVE_THRESHOLD, InfectionSummaryAboveThresholdDone)
_register_done(DoneFunctionType.NO_MORE_INFECTIONS, NoMoreInfectionsDone)
_register_done(DoneFunctionType.NO_PANDEMIC, NoPandemicDone)

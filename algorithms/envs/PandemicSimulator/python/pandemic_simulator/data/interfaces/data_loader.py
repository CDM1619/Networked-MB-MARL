# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import dataclasses
from abc import ABC
from typing import List, Optional, Sequence

import numpy as np

from ...environment import PandemicObservation, PandemicSimOpts

__all__ = ['ExperimentResult', 'ExperimentDataLoader', 'StageSchedule']


@dataclasses.dataclass(frozen=True)
class StageSchedule:
    stage: int
    end_day: Optional[int] = None


@dataclasses.dataclass
class ExperimentResult:
    sim_opts: PandemicSimOpts
    seeds: List[Optional[int]]
    obs_trajectories: PandemicObservation  # TNC array with size_n = num_seeds, size_t = episode_length
    reward_trajectories: np.ndarray  # TNC array with size_n = num_seeds, size_t = episode_length
    strategy: Sequence[StageSchedule]
    num_persons: int


class ExperimentDataLoader(ABC):
    """An interface for a data loader."""

    def get_data(self) -> Sequence[ExperimentResult]:
        """Return data as a sequence of ExperimentResult instances"""
        pass

# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import dataclasses
from collections import OrderedDict
from pathlib import Path
from typing import Sequence, Set, Tuple, Any

import h5py as h5
import numpy as np

from pandemic_simulator.environment import PandemicObservation
from .interfaces import ExperimentDataLoader, ExperimentResult, StageSchedule
from ..environment import PandemicSimOpts

__all__ = ['H5DataLoader']


class H5DataLoader(ExperimentDataLoader):
    """Implement a H5 experiment data loader"""

    _filename: Path
    _pandemic_sim_opts_field_names: Set[str]

    def __init__(self, filename: str, path: Path = Path('.'), ) -> None:
        """
        :param filename: filename
        :param path: path to store the h5 dataset
        """
        self._filename = path / filename
        self._pandemic_sim_opts_field_names = {f.name for f in dataclasses.fields(PandemicSimOpts)}

    def get_data(self) -> Sequence[ExperimentResult]:
        res = OrderedDict()

        with h5.File(self._filename, mode='r') as f:
            for trial_key in f.keys():
                group = f[trial_key]
                sim_opts_data = dict()

                exp_id = group.attrs.get('exp_id', None)

                seed = group.attrs.get('seed', None)
                num_persons = group.attrs.get('num_persons', None)
                num_stages_to_execute = group.attrs.get('num_stages_to_execute', None)

                # back compatibility with previously saved data
                if num_stages_to_execute is None:
                    strategy: Tuple[StageSchedule, ...] = (StageSchedule(
                        stage=group.attrs.get('stage_to_execute'), end_day=None),)
                else:
                    strategy = tuple([
                        StageSchedule(
                            stage=group.attrs.get(f'stage_{i}')[0],
                            end_day=None if group.attrs.get(f'stage_{i}')[1] == -1 else group.attrs.get(f'stage_{i}')[1]
                        )
                        for i in range(num_stages_to_execute)
                    ])

                for k, v in group.attrs.items():
                    if k in self._pandemic_sim_opts_field_names:
                        sim_opts_data[k] = tuple(v) if isinstance(v, np.ndarray) else v

                pandemic_obs = {k: v[:] for k, v in group['observation'].items()}
                rewards = np.atleast_3d(group['reward'][:])

                sim_opts = PandemicSimOpts(**sim_opts_data)

                # back compatibility with previously saved data
                if exp_id is None:
                    key: Tuple[Any, ...] = (sim_opts,)
                    key += strategy + (num_persons,)
                else:
                    key = exp_id

                if key not in res:
                    res[key] = ExperimentResult(sim_opts=sim_opts,
                                                seeds=[seed],
                                                obs_trajectories=PandemicObservation(**pandemic_obs),
                                                reward_trajectories=rewards,
                                                strategy=strategy,
                                                num_persons=num_persons)
                else:
                    res[key].seeds.append(seed)

                    for k, v in pandemic_obs.items():
                        pandemic_obs[k] = np.hstack((getattr(res[key].obs_trajectories, k), v))

                    res[key].obs_trajectories = PandemicObservation(**pandemic_obs)
                    res[key].reward_trajectories = np.hstack((res[key].reward_trajectories, rewards))

        return list(res.values())

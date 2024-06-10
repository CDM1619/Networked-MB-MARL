# Confidential, Copyright 2020, Sony Corporation of America, All rights reserved.

import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py as h5
import numpy as np

from .interfaces import ExperimentDataSaver
from ..environment import PandemicObservation
from ..utils import shallow_asdict

__all__ = ['H5DataSaver']


class H5DataSaver(ExperimentDataSaver):
    """Implement a H5 experiment data saver"""

    _filename: Path
    _f: h5.File
    _obs: Dict[str, np.ndarray]
    _rewards: Optional[np.ndarray]

    def __init__(self, filename: str, path: Path = Path('.'), overwrite: bool = False) -> None:
        """
        :param filename: filename
        :param path: path to store the h5 dataset
        :param overwrite: set to True to overwrite the dataset if one exists already at the specified path
        """
        self._filename = path / filename
        if self._filename.exists() and not overwrite:
            raise ValueError(f'{self._filename} already exists! Specify a new path or set overwrite=True')

        self._f = h5.File(self._filename, mode='w')
        self._obs = dict()
        self._rewards = None

    def begin(self, obs: PandemicObservation) -> None:
        self._obs = dict(**shallow_asdict(obs))
        self._rewards = None

    def record(self, obs: PandemicObservation, reward: Optional[Union[np.ndarray, float]] = None) -> None:
        for k, v in shallow_asdict(obs).items():
            self._obs[k] = np.vstack((self._obs[k], v))

        if reward is not None:
            reward = np.array(reward)
            self._rewards = reward if self._rewards is None else np.vstack((self._rewards, reward))

    def finalize(self, **kwargs: Any) -> bool:
        if not np.any(self._obs['infection_above_threshold']):
            # skip since infection never went about threshold
            return False

        g = self._f.create_group(time.strftime('%Y-%m-%dT%H:%M:%SZ'))

        g.attrs.update(**kwargs)
        obs = g.create_group('observation')

        for k, v in self._obs.items():
            obs.create_dataset(k, data=v.astype('float32') if v.dtype == 'O' else v)

        if self._rewards is not None:
            g.create_dataset('reward', data=self._rewards)

        self._f.flush()
        return True

    def close(self) -> None:
        self._f.close()
